from __future__ import division

import os

import numpy as np
from numpy.testing import assert_allclose
import pytest

import radical.residuals


@pytest.mark.parametrize(
    'Ax, Ay, x, y',
    [
        (3, 5, 0.0003, -0.0005),
        (30, 25, -0.001, 0.0008),
    ]
)
def test_firstorder(mockms, Ax, Ay, x, y):
    U, V = mockms.U, mockms.V
    model = np.ones((1, mockms.data.shape[0], 1), dtype=np.complex128)

    phases = x * U + y * V
    phases = phases[mockms.ANTENNA1] - phases[mockms.ANTENNA2]

    data = np.ones((mockms.data.shape[0], 1, 2), dtype=np.complex128)
    data = np.array([Ax, Ay])[None, None, :] * data * np.exp(-1j * phases)[:, None, None]

    res = radical.residuals.full_firstorder(np.array([Ax, Ay, x, y]), U, V, mockms.ant1, mockms.ant2, data, model)

    desired = np.zeros_like(res)
    assert_allclose(res, desired, atol=1e-7)


@pytest.mark.parametrize(
    'Ax, Ay, x, y, xx, xy, yy',
    [
        (3, 5, 0.0003, -0.0005, 1e-6, 2.5e-7, 15e-6),
        (30, 25, -0.001, 0.0008, -1.5e7, 2.3e6, -2e-5),
    ]
)
def test_secondorder(mockms, Ax, Ay, x, y, xx, xy, yy):
    U, V = mockms.U, mockms.V
    model = np.ones((1, mockms.data.shape[0], 1), dtype=np.complex128)

    phases = x * U + y * V + xx * U**2 + xy * U * V + yy * V**2
    phases = phases[mockms.ant1] - phases[mockms.ant2]

    data = np.ones((mockms.data.shape[0], 1, 2), dtype=np.complex128)
    data = np.array([Ax, Ay])[None, None, :] * data * np.exp(-1j * phases)[:, None, None]

    res = radical.residuals.full_secondorder(np.array([Ax, Ay, x, y, xx, xy, yy]), U, V, mockms.ant1, mockms.ant2, data, model)

    desired = np.zeros_like(res)
    assert_allclose(res, desired, atol=1e-7)


@pytest.mark.parametrize(
    'A1x, A1y, A2x, A2y, x, y',
    [
        (3, 5, 2, 3, 0.0003, -0.0005),
        (30, 25, 1, 4, -0.001, 0.0008),
    ]
)
def test_firstorder_multiple_models(mockms, A1x, A1y, A2x, A2y, x, y):
    U, V = mockms.U, mockms.V
    u, v, w = mockms.uvw.T
    models = np.empty((2, mockms.data.shape[0], 1), dtype=np.complex128)

    models[0, :, 0] = np.exp(2j * np.pi * (u*0.5 + v*0.2 + w*(1 - 0.5**2 - 0.2**2) - 1))
    models[1, :, 0] = np.exp(2j * np.pi * (u*-0.1 + v*0.4 + w*(1 - (-0.1)**2 - 0.4**2) - 1))

    phases = x * U + y * V
    phases = phases[mockms.ant1] - phases[mockms.ant2]

    data = np.zeros((mockms.data.shape[0], 1, 2), dtype=np.complex128)
    data[:, :, 0] += A1x * models[0]
    data[:, :, 1] += A1y * models[0]
    data[:, :, 0] += A2x * models[1]
    data[:, :, 1] += A2y * models[1]
    data *= np.exp(-1j * phases)[:, None, None]

    res = radical.residuals.full_firstorder(np.array([A1x, A1y, A2x, A2y, x, y]), U, V, mockms.ant1, mockms.ant2, data, models)

    desired = np.zeros_like(res)
    assert_allclose(res, desired, atol=1e-7)


@pytest.mark.parametrize(
    'A1x, A1y, A2x, A2y, x, y, xx, xy, yy',
    [
        (3, 5, 2, 3, 0.0003, -0.0005, 1e-6, 2.5e-7, 15e-6),
        (30, 25, 1, 4, -0.001, 0.0008, -1.5e7, 2.3e6, -2e-5),
    ]
)
def test_secondorder_multiple_models(mockms, A1x, A1y, A2x, A2y, x, y, xx, xy, yy):
    U, V = mockms.U, mockms.V
    u, v, w = mockms.uvw.T
    models = np.empty((2, mockms.data.shape[0], 1), dtype=np.complex128)

    models[0, :, 0] = np.exp(2j * np.pi * (u*0.5 + v*0.2 + w*(1 - 0.5**2 - 0.2**2) - 1))
    models[1, :, 0] = np.exp(2j * np.pi * (u*-0.1 + v*0.4 + w*(1 - (-0.1)**2 - 0.4**2) - 1))

    phases = x * U + y * V + xx * U**2 + xy * U * V + yy * V**2
    phases = phases[mockms.ant1] - phases[mockms.ant2]

    data = np.zeros((mockms.data.shape[0], 1, 2), dtype=np.complex128)
    data[:, :, 0] += A1x * models[0]
    data[:, :, 1] += A1y * models[0]
    data[:, :, 0] += A2x * models[1]
    data[:, :, 1] += A2y * models[1]
    data *= np.exp(-1j * phases)[:, None, None]

    res = radical.residuals.full_secondorder(np.array([A1x, A1y, A2x, A2y, x, y, xx, xy, yy]), U, V, mockms.ant1, mockms.ant2, data, models)

    desired = np.zeros_like(res)
    assert_allclose(res, desired, atol=1e-7)
