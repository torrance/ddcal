from __future__ import print_function, division

from numba import njit, int32, float64, complex128, prange
import numpy as np


@njit([float64[:](float64[:], float64[:], float64[:], int32[:], int32[:], complex128[:, :, :], complex128[:, :, :])], parallel=True)
def full_firstorder(params, U, V, ant1, ant2, data, models):
    amps = params[:-2]
    x, y = params[-2:]

    phases = x * U + y * V

    residuals = np.empty_like(data)
    for row in prange(0, data.shape[0]):
        phase = np.exp(1j * (phases[ant1[row]] - phases[ant2[row]]))
        residuals[row, :, 0] = phase * data[row, :, 0]
        residuals[row, :, 1] = phase * data[row, :, 1]

        for i, (Ax, Ay) in enumerate(zip(amps[0::2], amps[1::2])):
            residuals[row, :, 0] -= Ax * models[i, row]
            residuals[row, :, 1] -= Ay * models[i, row]

    residuals = residuals.flatten()
    residuals = residuals[np.isfinite(residuals)]
    residuals = np.concatenate((residuals.real, residuals.imag))

    return residuals


@njit([float64[:](float64[:], float64[:], float64[:], int32[:], int32[:], complex128[:, :, :], complex128[:, :, :])], parallel=True)
def full_secondorder(params, U, V, ant1, ant2, data, models):
    amps = params[:-5]
    x, y, xx, xy, yy = params[-5:]

    phases = x * U + y * V + xx * U**2 + xy * U * V + yy * V**2

    residuals = np.empty_like(data)
    for row in prange(0, data.shape[0]):
        phase = np.exp(1j * (phases[ant1[row]] - phases[ant2[row]]))
        residuals[row, :, 0] = phase * data[row, :, 0]
        residuals[row, :, 1] = phase * data[row, :, 1]

        for i, (Ax, Ay) in enumerate(zip(amps[0::2], amps[1::2])):
            residuals[row, :, 0] -= Ax * models[i, row]
            residuals[row, :, 1] -= Ay * models[i, row]

    residuals = residuals.flatten()
    residuals = residuals[np.isfinite(residuals)]
    residuals = np.concatenate((residuals.real, residuals.imag))

    return residuals
