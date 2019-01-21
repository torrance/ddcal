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
    u, v, w = mockms.u_lambda, mockms.v_lambda, mockms.w_lambda
    phases = np.zeros(len(u))

    mockms.data = np.array([Ax, 0, 0, Ay]) * np.exp(
        2j * np.pi * (u * l + v * m + w * (np.sqrt(1 - l**2 - m**2) - 1))
    )[:, :, None]

    radical.peel.peel(mockms.data, u, v, w, l, m, Ax, Ay, phases)

    desired = np.zeros_like(mockms.data)
    assert_allclose(mockms.data, desired)


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
    phases = phases[mockms.ant1] - phases[mockms.ant2]

    u, v, w = mockms.u_lambda, mockms.v_lambda, mockms.w_lambda

    mockms.data = np.array([Ax, 0, 0, Ay]) * np.exp(
        2j * np.pi * (u * l + v * m + w * (np.sqrt(1 - l**2 - m**2) - 1))
    )[:, :, None]

    # Distort data  by random phases
    mockms.data *= np.exp(-1j * phases)[:, None, None]

    radical.peel.peel(mockms.data, u, v, w, l, m, Ax, Ay, phases)

    desired = np.zeros_like(mockms.data)
    assert_allclose(mockms.data, desired, atol=1e-7)


@pytest.mark.parametrize(
    "Ax, Ay, l, m",
    [
        (5, 3, 0.1, -0.03),
        (100, 77, 0.01, 0.1),
        (1, 5, 0, 0),
    ]
)
def test_unpeel_and_peel(mockms, Ax, Ay, l, m):
    phases = np.random.uniform(-np.pi, np.pi, len(mockms.antids))
    phases = phases[mockms.ant1] - phases[mockms.ant2]

    u, v, w = mockms.u_lambda, mockms.v_lambda, mockms.w_lambda

    radical.peel.unpeel(mockms.data, u, v, w, l, m, Ax, Ay, phases)
    radical.peel.peel(mockms.data, u, v, w, l, m, Ax, Ay, phases)

    desired = np.zeros_like(mockms.data)
    assert_allclose(mockms.data, desired, atol=1e-7)
