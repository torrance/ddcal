from numba import njit, float64, complex128, prange
import numpy as np


@njit([(complex128[:, :, :], float64[:, :], float64[:, :], float64[:, :], float64, float64, float64, float64, float64[:])], parallel=True)
def peel(data, u, v, w, l, m, Ax, Ay, phases):
    n = np.sqrt(1 - l**2 - m**2) - 1

    for row in prange(0, u.shape[0]):
        phase = 2j * np.pi * (u[row]*l + v[row]*m + w[row]*n)
        phase = np.exp(phase - 1j * phases[row])

        data[row, :, 0] -= Ax * phase
        data[row, :, 3] -= Ay * phase


@njit([(complex128[:, :, :], float64[:, :], float64[:, :], float64[:, :], float64, float64, float64, float64, float64[:])], parallel=True)
def unpeel(data, u, v, w, l, m, Ax, Ay, phases):
    n = np.sqrt(1 - l**2 - m**2) - 1

    for row in prange(0, u.shape[0]):
        phase = 2j * np.pi * (u[row]*l + v[row]*m + w[row]*n)
        phase = np.exp(phase - 1j * phases[row])

        data[row, :, 0] += Ax * phase
        data[row, :, 3] += Ay * phase
