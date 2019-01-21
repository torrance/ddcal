from __future__ import print_function, division

import logging
import time as tm

from casacore.measures import measures
from casacore.quanta import quantity
from casacore.tables import taql
from numba import njit, float64, complex64, complex128, prange
import numpy as np

import radical.constants as constants


logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')


def phase_rotate(uvw, data, ra, dec, ra0, dec0, lambdas):
    # Calculate rotated uvw values
    start = tm.time()
    new_uvw = rotateuvw(uvw, ra, dec, ra0, dec0)
    elapsed = tm.time() - start
    logger.debug("Phase rotated uvw elapsed: %g", elapsed)

    # Calculate phase offset
    start = tm.time()
    new_data = woffset(data, uvw.T[2], new_uvw.T[2], lambdas)
    elapsed = tm.time() - start
    logger.debug("Phase rotated visibilities elapsed: %g", elapsed)

    return new_uvw, new_data


def rotateuvw(uvw, ra, dec, ra0, dec0):
    """
    We calculate new uvw values based on existing uvw values. Whilst this has the effect
    of propagating any existing uvw errors, it has the benefit of being mathematically
    self-consistent.

    Adopted from matrix equation 4.1, in Thompson, Moran, Swenson (3rd edition).
    Let (uvw) = r(ra, dec) * (xyz), then this formula is: r(ra, dec) * r^-1(ra0, dec0)
    """
    u, v, w = uvw.T
    uvwprime = np.empty_like(uvw)

    uvwprime[:, 0] = (
        u * np.cos(ra - ra0)
        + v * np.sin(dec0) * np.sin(ra - ra0)
        - w * np.cos(dec0) * np.sin(ra - ra0)
    )
    uvwprime[:, 1] = (
        -u * np.sin(dec) * np.sin(ra - ra0)
        + v * (np.sin(dec0) * np.sin(dec) * np.cos(ra - ra0) + np.cos(dec0) * np.cos(dec))
        + w * (np.sin(dec0) * np.cos(dec) - np.cos(dec0) * np.sin(dec) * np.cos(ra - ra0))
    )
    uvwprime[:, 2] = (
        u * np.cos(dec) * np.sin(ra - ra0)
        + v * (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0))
        + w * (np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0))
    )
    return uvwprime


@njit([complex128[:, :, :](complex128[:, :, :], float64[:], float64[:], float64[:])], parallel=True)
def woffset(data, oldw, neww, lambdas):
    offset = -2j * np.pi * (neww - oldw)
    phase = np.empty_like(data)
    for row in prange(0, phase.shape[0]):
        tmp = offset[row] / lambdas
        for pol in range(0, data.shape[2]):
            phase[row, :, pol] = tmp

    return data * np.exp(phase)
