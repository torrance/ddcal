from __future__ import division

import numpy as np
from numpy.testing import assert_allclose
import pytest

import radical.peel


@pytest.mark.parametrize(
    "Ax, Ay, l, m",
    [
        (5, 3, 0.1, -0.03),
        (100, 77, 0.01, 0.1),
        (1, 5, 0, 0),
    ]
)
def test_peel_with_zero_phases(mockms, Ax, Ay, l, m):
    u, v, w = mockms.UVW.T[:, :, None] / mockms.CHAN_FREQ
    phases = np.zeros(u.shape[0])

    data = np.array([Ax, 0, 0, Ay]) * np.exp(
        2j * np.pi * (u * l + v * m + w * (np.sqrt(1 - l**2 - m**2) - 1))
    )[:, :, None]

    radical.peel.peel(data, u, v, w, l, m, Ax, Ay, phases)

    desired = np.zeros_like(data)
    assert_allclose(data, desired)


@pytest.mark.parametrize(
    "Ax, Ay, l, m",
    [
        (5, 3, 0.1, -0.03),
        (100, 77, 0.01, 0.1),
        (1, 5, 0, 0),
    ]
)
def test_peel_with_random_phases(mockms, Ax, Ay, l, m):
    phases = np.random.uniform(-np.pi, np.pi, 128)
    phases = phases[mockms.ANTENNA1] - phases[mockms.ANTENNA2]

    u, v, w = mockms.UVW.T[:, :, None] / mockms.CHAN_FREQ

    data = np.array([Ax, 0, 0, Ay]) * np.exp(
        2j * np.pi * (u * l + v * m + w * (np.sqrt(1 - l**2 - m**2) - 1))
    )[:, :, None]

    # Distort data  by random phases
    data *= np.exp(-1j * phases)[:, None, None]

    radical.peel.peel(data, u, v, w, l, m, Ax, Ay, phases)

    desired = np.zeros_like(data)
    assert_allclose(data, desired, atol=1e-7)


@pytest.mark.parametrize(
    "Ax, Ay, l, m",
    [
        (5, 3, 0.1, -0.03),
        (100, 77, 0.01, 0.1),
        (1, 5, 0, 0),
    ]
)
def test_unpeel_and_peel(mockms, Ax, Ay, l, m):
    phases = np.random.uniform(-np.pi, np.pi, 128)
    phases = phases[mockms.ANTENNA1] - phases[mockms.ANTENNA2]

    u, v, w = mockms.UVW.T[:, :, None] / mockms.CHAN_FREQ

    data = np.zeros((u.shape[0], u.shape[1], 4), dtype=np.complex128)

    radical.peel.unpeel(data, u, v, w, l, m, Ax, Ay, phases)
    radical.peel.peel(data, u, v, w, l, m, Ax, Ay, phases)

    desired = np.zeros_like(data)
    assert_allclose(data, desired, atol=1e-7)
