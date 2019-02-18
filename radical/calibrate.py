from __future__ import print_function, division

import logging
import time as tm

from numba import njit, prange, complex128
import numpy as np
from scipy.optimize import least_squares

from radical.coordinates import radec_to_lm
from radical.phaserotate import phase_rotate
import radical.residuals as residuals


logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')


def solve(src, solution, mset, order):
    # Phase rotate onto source and average in frequency
    uvw, rotated = phase_rotate(mset.uvw, mset.data[:, :, [True, False, False, True]], src.ra, src.dec, mset.ra0, mset.dec0, mset.lambdas)

    start = tm.time()
    rotated = freq_average(rotated)[:, None, :]
    elapsed = tm.time() - start
    logger.debug("Frequency averaging elapsed: %g", elapsed)

    # Create array of unscaled (flux = 1) point sources for each component
    start = tm.time()
    u_lambda, v_lambda, w_lambda = uvw.T[:, :, None] / mset.midlambda
    models = np.empty((len(src.components), rotated.shape[0], rotated.shape[1]), dtype=np.complex128)
    for i, comp in enumerate(src.components):
        l, m = radec_to_lm(comp.ra, comp.dec, src.ra, src.dec)
        models[i] = np.exp(2j * np.pi * (
            u_lambda*l + v_lambda*m + w_lambda*(np.sqrt(1 - l**2 - m**2) - 1)
        ))
    elapsed = tm.time() - start
    logger.debug("Model creation elapsed: %g", elapsed)

    # Fit
    logger.debug("Fitting source '%s'...", src.name)
    if order == 1:
        f = residuals.full_firstorder
    elif order == 2:
        f = residuals.full_secondorder

    start = tm.time()
    res = least_squares(
        f,
        x0=solution.get_params(order=order),
        args=(mset.U, mset.V, mset.ant1, mset.ant2, rotated, models),
        verbose=1,
        x_scale=solution.x_scale(order=order),
    )
    logger.debug("Fit (order=%d) elapsed: %g", order, tm.time() - start)
    logger.debug(res.message)
    logger.debug("Fit params:" + " %g" * len(res.x), *res.x)

    # If fit converged, add solution or else mark it as failed
    if res.success:
        solution.set_params(res.x)
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


def reduced_chi_squared(squaredsum, nsamples, nparams, sigma):
    chi_squared = squaredsum / sigma**2
    return chi_squared / (nsamples - nparams)
