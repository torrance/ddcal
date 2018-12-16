from __future__ import print_function, division

from numba import njit, int32, float64, complex128, prange
import numpy as np


@njit([float64[:](float64[:], float64[:], float64[:], int32[:], int32[:], complex128[:, :, :], complex128[:, :, :])], parallel=True)
def full_firstorder(params, U, V, ant1, ant2, data, model):
    Ax, Ay, x, y = params

    phases = x * U + y * V

    residuals = np.empty_like(data)
    for row in prange(0, model.shape[0]):
        phase = np.exp(1j * (phases[ant1[row]] - phases[ant2[row]]))
        residuals[row, :, 0] = phase * data[row, :, 0] - Ax * model[row, :, 0]
        residuals[row, :, 1] = phase * data[row, :, 1] - Ay * model[row, :, 1]

    residuals = residuals.flatten()
    residuals = residuals[~np.isnan(residuals)]
    residuals = np.concatenate((residuals.real, residuals.imag))

    return residuals


@njit([float64[:](float64[:], float64[:], float64[:], int32[:], int32[:], complex128[:, :, :], complex128[:, :, :])], parallel=True)
def full_secondorder(params, U, V, ant1, ant2, data, model):
    Ax, Ay, x, y, xx, xy, yy = params

    phases = x * U + y * V + xx * U**2 + xy * U * V + yy * V**2

    residuals = np.empty_like(data)
    for row in prange(0, model.shape[0]):
        phase = np.exp(1j * (phases[ant1[row]] - phases[ant2[row]]))
        residuals[row, :, 0] = phase * data[row, :, 0] - Ax * model[row, :, 0]
        residuals[row, :, 1] = phase * data[row, :, 1] - Ay * model[row, :, 1]

    residuals = residuals.flatten()
    residuals = residuals[~np.isnan(residuals)]
    residuals = np.concatenate((residuals.real, residuals.imag))

    return residuals
