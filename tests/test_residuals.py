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
    u, v, w = mockms.UVW.T[:, :, None] / mockms.CHAN_FREQ
    U, V = mockms.U, mockms.V

    model = np.ones((u.shape[0], u.shape[1], 2), dtype=np.complex128)

    phases = x * mockms.U + y * mockms.V
    phases = phases[mockms.ANTENNA1] - phases[mockms.ANTENNA2]

    data = np.array([Ax, Ay])[None, None, :] * model * np.exp(-1j * phases)[:, None, None]

    res = radical.residuals.full_firstorder(np.array([Ax, Ay, x, y]), U, V, mockms.ANTENNA1, mockms.ANTENNA2, data, model)

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
    u, v, w = mockms.UVW.T[:, :, None] / mockms.CHAN_FREQ
    U, V = mockms.U, mockms.V

    model = np.ones((u.shape[0], u.shape[1], 2), dtype=np.complex128)

    phases = x * U + y * V + xx * U**2 + xy * U * V + yy * V**2
    phases = phases[mockms.ANTENNA1] - phases[mockms.ANTENNA2]

    data = np.array([Ax, Ay])[None, None, :] * model * np.exp(-1j * phases)[:, None, None]

    res = radical.residuals.full_secondorder(np.array([Ax, Ay, x, y, xx, xy, yy]), U, V, mockms.ANTENNA1, mockms.ANTENNA2, data, model)

    desired = np.zeros_like(res)
    assert_allclose(res, desired, atol=1e-7)
