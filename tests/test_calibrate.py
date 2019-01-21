from __future__ import division

from astropy.coordinates import SkyCoord
import astropy.units as unit
import numpy as np
from numpy.testing import assert_allclose
import pytest

import radical.calibrate as calibrate
import radical.constants as constants
from radical.coordinates import radec_to_lm
import radical.phaserotate as phaserotate
from radical.skymodel import Component
from radical.solution import Solution


@pytest.mark.parametrize(
    'Ax, Ay',
    [
        (4, 3.5),
        (1.4, 2.7),
        (20, 47),
    ]
)
def test_firstorder_amplitudefit_centered(mockms, mockcomp, Ax, Ay):
    u, v, w = mockms.u_lambda, mockms.v_lambda, mockms.w_lambda
    l, m = radec_to_lm(mockms.ra0, mockms.dec0, mockms.ra0, mockms.dec0)

    mockms.data = np.array([Ax, 0, 0, Ay]) * np.exp(
        2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1))
    )[:, :, None]

    solution = Solution()

    mockcomp.ra = mockms.ra0
    mockcomp.dec = mockms.dec0

    calibrate.solve(mockcomp, solution, mockms, 1)
    params = solution.get_params(2)

    assert_allclose(params, [Ax, Ay, 0, 0, 0, 0, 0], atol=1e-15)


@pytest.mark.parametrize(
    'Ax, Ay, ra, dec',
    [
        (4, 3.5, 0.19, -0.32),
    ]
)
def test_firstorder_amplitudefit(mockms, mockcomp, Ax, Ay, ra, dec):
    u, v, w = mockms.u_lambda, mockms.v_lambda, mockms.w_lambda
    l, m = radec_to_lm(ra, dec, mockms.ra0, mockms.dec0)

    mockms.data = np.array([Ax, 0, 0, Ay]) * np.exp(
        2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1))
    )[:, :, None]

    solution = Solution()

    mockcomp.ra = ra
    mockcomp.dec = dec

    calibrate.solve(mockcomp, solution, mockms, 1)
    params = solution.get_params(2)

    assert_allclose(params[:2], [Ax, Ay], rtol=1e-2)
    assert_allclose(params[2:4], [0, 0], atol=1e-7)
    assert_allclose(params[4:], [0, 0, 0])


@pytest.mark.parametrize(
    'Ax, Ay, ra, dec',
    [
        (4, 3.5, 0.19, -0.32),
    ]
)
def test_firstorder__multiplesources_amplitudefit(mockms, mockcomp, Ax, Ay, ra, dec):
    u, v, w = mockms.u_lambda, mockms.v_lambda, mockms.w_lambda
    l, m = radec_to_lm(ra, dec, mockms.ra0, mockms.dec0)

    mockms.data = np.array([Ax, 0, 0, Ay]) * np.exp(
        2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1))
    )[:, :, None]

    # Add a source at l=0, m=0
    # In part, this tests that we are filtering out autocorrelations
    mockms.data[:, :, 0] += Ax
    mockms.data[:, :, 3] += Ay

    solution = Solution()

    mockcomp.ra = ra
    mockcomp.dec = dec

    calibrate.solve(mockcomp, solution, mockms, 1)
    params = solution.get_params(2)

    assert_allclose(params[:2], [Ax, Ay], rtol=1e-2)
    assert_allclose(params[2:4], [0, 0], atol=5e-6)
    assert_allclose(params[4:], [0, 0, 0])


@pytest.mark.parametrize(
    'Ax, Ay, x, y',
    [
        (4, 3.5, 3e-5, -3.5e-5),
    ]
)
def test_firstorder_centered(mockms, mockcomp, Ax, Ay, x, y):
    mockms.data[:, :, 0] = Ax
    mockms.data[:, :, 3] = Ay

    phases = x * mockms.U + y * mockms.V
    phases = phases[mockms.ant1] - phases[mockms.ant2]
    mockms.data *= np.exp(-1j * phases)[:, None, None]

    mockcomp.ra = mockms.ra0
    mockcomp.dec = mockms.dec0

    solution = Solution()
    calibrate.solve(mockcomp, solution, mockms, 1)
    params = solution.get_params(2)

    assert_allclose(params[:2], [Ax, Ay], rtol=1e-2)
    assert_allclose(params[2:4], [x, y], atol=5e-6)
    assert_allclose(params[4:], [0, 0, 0])


@pytest.mark.parametrize(
    'Ax, Ay, ra, dec, x, y',
    [
        (4, 3.5, 0.19, -0.32, 3e-5, -3.5e-5),
    ]
)
def test_firstorder(mockms, mockcomp, Ax, Ay, ra, dec, x, y):
    u, v, w = mockms.u_lambda, mockms.v_lambda, mockms.w_lambda
    l, m = radec_to_lm(ra, dec, mockms.ra0, mockms.dec0)

    mockms.data = np.array([Ax, 0, 0, Ay]) * np.exp(
        2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1))
    )[:, :, None]

    phases = x * mockms.U + y * mockms.V
    phases = phases[mockms.ant1] - phases[mockms.ant2]
    mockms.data *= np.exp(-1j * phases)[:, None, None]

    mockcomp.ra = ra
    mockcomp.dec = dec

    solution = Solution()
    calibrate.solve(mockcomp, solution, mockms, 1)
    params = solution.get_params(2)

    assert_allclose(params[:2], [Ax, Ay], rtol=1e-2)
    assert_allclose(params[2:4], [x, y], atol=5e-6)
    assert_allclose(params[4:], [0, 0, 0])


@pytest.mark.parametrize(
    'Ax, Ay, ra, dec, x, y',
    [
        (4, 3.5, 0.19, -0.32, 3e-5, -3.5e-5),
    ]
)
def test_firstorder_with_noise(mockms, mockcomp, Ax, Ay, ra, dec, x, y):
    u, v, w = mockms.u_lambda, mockms.v_lambda, mockms.w_lambda
    l, m = radec_to_lm(ra, dec, mockms.ra0, mockms.dec0)

    mockms.data = np.array([Ax, 0, 0, Ay]) * np.exp(
        2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1))
    )[:, :, None]

    phases = x * mockms.U + y * mockms.V
    phases = phases[mockms.ant1] - phases[mockms.ant2]
    mockms.data *= np.exp(-1j * phases)[:, None, None]

    mockms.data += np.random.normal(0, 40, mockms.data.shape) + 1j *np.random.normal(0, 40, mockms.data.shape)

    mockcomp.ra = ra
    mockcomp.dec = dec

    solution = Solution()
    calibrate.solve(mockcomp, solution, mockms, 1)
    params = solution.get_params(2)

    assert_allclose(params[:2], [Ax, Ay], rtol=1e-2)
    assert_allclose(params[2:4], [x, y], atol=5e-6)
    assert_allclose(params[4:], [0, 0, 0])


@pytest.mark.parametrize(
    'Ax, Ay, ra, dec, x, y',
    [
        (4, 3.5, 0.19, -0.32, 3e-5, -3.5e-5),
    ]
)
def test_firstorder_multiple_sources_with_noise(mockms, mockcomp, Ax, Ay, ra, dec, x, y):
    u, v, w = mockms.u_lambda, mockms.v_lambda, mockms.w_lambda
    l, m = radec_to_lm(ra, dec, mockms.ra0, mockms.dec0)

    mockms.data = np.array([Ax, 0, 0, Ay]) * np.exp(
        2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1))
    )[:, :, None]

    # Additional sources
    l, m = 0, 0.05
    mockms.data += np.array([Ax, 0, 0, Ay]) * np.exp(
        2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1))
    )[:, :, None]
    l, m = 0.6, -0.5
    mockms.data += np.array([Ax, 0, 0, Ay]) * np.exp(
        2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1))
    )[:, :, None]

    phases = x * mockms.U + y * mockms.V
    phases = phases[mockms.ant1] - phases[mockms.ant2]
    mockms.data *= np.exp(-1j * phases)[:, None, None]

    mockms.data += np.random.normal(0, 40, mockms.data.shape) + 1j *np.random.normal(0, 40, mockms.data.shape)

    mockcomp.ra = ra
    mockcomp.dec = dec

    solution = Solution()
    calibrate.solve(mockcomp, solution, mockms, 1)
    params = solution.get_params(2)

    assert_allclose(params[:2], [Ax, Ay], rtol=1e-2)
    assert_allclose(params[2:4], [x, y], atol=5e-6)
    assert_allclose(params[4:], [0, 0, 0])


@pytest.mark.parametrize(
    'Ax, Ay, ra, dec, x, y, xx, xy, yy',
    [
        (4, 3.5, 0.19, -0.32, 3e-5, -3.5e-5, 1.4e-7, -1.9e-8, 7e-8),
    ]
)
def test_secondorder(mockms, mockcomp, Ax, Ay, ra, dec, x, y, xx, xy, yy):
    u, v, w = mockms.u_lambda, mockms.v_lambda, mockms.w_lambda
    l, m = radec_to_lm(ra, dec, mockms.ra0, mockms.dec0)

    mockms.data = np.array([Ax, 0, 0, Ay]) * np.exp(
        2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1))
    )[:, :, None]

    U, V = mockms.U, mockms.V
    phases = x * U + y * V + xx * U**2 + xy * U * V + yy * V**2
    phases = phases[mockms.ant1] - phases[mockms.ant2]
    mockms.data *= np.exp(-1j * phases)[:, None, None]

    mockcomp.ra = ra
    mockcomp.dec = dec

    solution = Solution()
    calibrate.solve(mockcomp, solution, mockms, 1)
    calibrate.solve(mockcomp, solution, mockms, 2)
    params = solution.get_params(2)

    assert_allclose(params[:2], [Ax, Ay], rtol=1e-2)
    assert_allclose(params[2:4], [x, y], atol=5e-6)
    assert_allclose(params[4:], [xx, xy, yy], atol=5e-9)


@pytest.mark.parametrize(
    'Ax, Ay, ra, dec, x, y, xx, xy, yy',
    [
        (4, 3.5, 0.19, -0.32, 3e-5, -3.5e-5, 1.4e-7, -1.9e-8, 7e-8),
    ]
)
def test_secondorder_multiple_sources(mockms, mockcomp, Ax, Ay, ra, dec, x, y, xx, xy, yy):
    u, v, w = mockms.u_lambda, mockms.v_lambda, mockms.w_lambda
    l, m = radec_to_lm(ra, dec, mockms.ra0, mockms.dec0)

    mockms.data = np.array([Ax, 0, 0, Ay]) * np.exp(
        2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1))
    )[:, :, None]

    # Additional sources
    l, m = 0, 0.05
    mockms.data += np.array([Ax, 0, 0, Ay]) * np.exp(
        2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1))
    )[:, :, None]
    l, m = 0.6, -0.5
    mockms.data += np.array([Ax, 0, 0, Ay]) * np.exp(
        2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1))
    )[:, :, None]

    U, V = mockms.U, mockms.V
    phases = x * U + y * V + xx * U**2 + xy * U * V + yy * V**2
    phases = phases[mockms.ant1] - phases[mockms.ant2]
    mockms.data *= np.exp(-1j * phases)[:, None, None]

    mockms.data += np.random.normal(0, 40, mockms.data.shape) + 1j *np.random.normal(0, 40, mockms.data.shape)

    mockcomp.ra = ra
    mockcomp.dec = dec

    solution = Solution()
    calibrate.solve(mockcomp, solution, mockms, 1)
    calibrate.solve(mockcomp, solution, mockms, 2)
    params = solution.get_params(2)

    assert_allclose(params[:2], [Ax, Ay], rtol=1e-2)
    assert_allclose(params[2:4], [x, y], atol=5e-6)
    assert_allclose(params[4:], [xx, xy, yy], atol=5e-9)


@pytest.mark.parametrize(
    'Ax, Ay, ra, dec, x, y, xx, xy, yy',
    [
        (4, 3.5, 0.19, -0.32, 3e-5, -3.5e-5, 1.4e-7, -1.9e-8, 7e-8),
    ]
)
def test_secondorder_multiple_sources(mockms, mockcomp, Ax, Ay, ra, dec, x, y, xx, xy, yy):
    u, v, w = mockms.u_lambda, mockms.v_lambda, mockms.w_lambda
    l, m = radec_to_lm(ra, dec, mockms.ra0, mockms.dec0)

    mockms.data = np.array([Ax, 0, 0, Ay]) * np.exp(
        2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1))
    )[:, :, None]

    # Additional sources
    l, m = 0, 0.05
    mockms.data += np.array([Ax, 0, 0, Ay]) * np.exp(
        2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1))
    )[:, :, None]
    l, m = 0.6, -0.5
    mockms.data += np.array([Ax, 0, 0, Ay]) * np.exp(
        2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1))
    )[:, :, None]

    U, V = mockms.U, mockms.V
    phases = x * U + y * V + xx * U**2 + xy * U * V + yy * V**2
    phases = phases[mockms.ant1] - phases[mockms.ant2]
    mockms.data *= np.exp(-1j * phases)[:, None, None]

    mockcomp.ra = ra
    mockcomp.dec = dec

    solution = Solution()
    calibrate.solve(mockcomp, solution, mockms, 1)
    calibrate.solve(mockcomp, solution, mockms, 2)
    params = solution.get_params(2)

    assert_allclose(params[:2], [Ax, Ay], rtol=1e-2)
    assert_allclose(params[2:4], [x, y], atol=5e-6)
    assert_allclose(params[4:], [xx, xy, yy], atol=5e-9)
