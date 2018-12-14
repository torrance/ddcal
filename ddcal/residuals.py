from __future__ import print_function, division

from numba import jit
import numpy as np


def full_firstorder((Ax, Ay, x, y), U, V, ant1, ant2, data, model):
    phases = x * U + y * V

    model = model.copy()
    model[:, :, 0] = Ax * model[:, :, 0]
    model[:, :, 1] = Ay * model[:, :, 1]

    corrected = data.copy()
    corrected[:, :, 0] = np.exp(1j * (phases[ant1][:, None] - phases[ant2][:, None])) * data[:, :, 0]
    corrected[:, :, 1] = np.exp(1j * (phases[ant1][:, None] - phases[ant2][:, None])) * data[:, :, 1]

    residual = corrected - model
    residual = residual[~np.isnan(residual)]
    residual = np.concatenate([residual.real, residual.imag])

    return residual


def full_secondorder((Ax, Ay, x, y, xx, xy, yy), U, V, ant1, ant2, data, model):
    phases = x * U + y * V + xx * U**2 + xy * U * V + yy * V**2

    model = model.copy()
    model[:, :, 0] = Ax * model[:, :, 0]
    model[:, :, 1] = Ay * model[:, :, 1]

    corrected = data.copy()
    corrected[:, :, 0] = np.exp(1j * (phases[ant1][:, None] - phases[ant2][:, None])) * data[:, :, 0]
    corrected[:, :, 1] = np.exp(1j * (phases[ant1][:, None] - phases[ant2][:, None])) * data[:, :, 1]

    residual = corrected - model
    residual = residual[~np.isnan(residual)]
    residual = np.concatenate([residual.real, residual.imag])

    return residual
