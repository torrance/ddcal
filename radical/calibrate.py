from __future__ import print_function, division

import logging
import time as tm

from numba import njit, prange, complex128
import numpy as np
from scipy.optimize import least_squares

from radical.phaserotate import phase_rotate
import radical.residuals as residuals


logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')


def solve(comp, solution, mset, order):
    # Phase rotate onto source and average in frequency
    _, rotated = phase_rotate(mset.uvw, mset.data[:, :, [True, False, False, True]], comp.ra, comp.dec, mset.ra0, mset.dec0, mset.lambdas)

    start = tm.time()
    rotated = freq_average(rotated)[:, None, :]
    elapsed = tm.time() - start
    logger.debug("Frequency averaging elapsed: %g", elapsed)

    # Create model: point source located at phase center
    model = np.ones_like(rotated)

    # Fit
    logger.debug(
        "Fitting source located at %s...",
        comp.position.to_string('hmsdms'),
    )
    if order == 1:
        f = residuals.full_firstorder
    elif order == 2:
        f = residuals.full_secondorder

    start = tm.time()
    res = least_squares(
        f,
        x0=solution.get_params(order=order),
        args=(mset.U, mset.V, mset.ant1, mset.ant2, rotated, model),
        verbose=2,
        x_scale=solution.x_scale(order=order),
    )
    logger.debug("Fit (order=%d) elapsed: %g", order, tm.time() - start)
    logger.debug(res.message)
    logger.debug(
        "Model flux: (%g, %g) versus fit flux (Ax Ay): %g %g",
        solution.Ax,
        solution.Ay,
        res.x[0],
        res.x[1],
    )
    logger.debug("Fit params:" + " %g" * len(res.x), *res.x)

    # If fit converged, add solution or else mark it as failed
    if res.success:
        solution.set_params(res.x)
        solution.snr = snr(rotated, solution, mset)
        logger.debug("SNR of fit: %g", solution.snr)
    else:
        logger.warning("Fit failed; marking solution as failed")
        solution.failed = True


@njit([complex128[:, :](complex128[:, :, :])], parallel=True)
def freq_average(data):
    averaged = np.empty_like(data[:, 0, :])
    for row in prange(0, data.shape[0]):
        for pol in prange(0, data.shape[2]):
            averaged[row, pol] = np.nanmean(data[row, :, pol])

    return averaged


def snr(data, solution, mset):
    start = tm.time()

    noise = data * np.exp(1j * solution.phasecorrections(mset))[:, None, None]
    noise[:, :, 0] -= solution.Ax
    noise[:, :, 1] -= solution.Ay
    noise = noise[np.isfinite(noise)]

    signal = 0.5 * (solution.Ax + solution.Ay) * len(noise)
    noise = abs(noise).sum()

    snr = signal / noise

    elapsed = tm.time() - start
    logger.debug("SNR calculation elapsed: %g",  elapsed)

    return snr
