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
    assert_allclose(params[4:], [0, 0, 0], atol=1e-11)


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
    assert_allclose(params[2:4], [0, 0], atol=1e-6)
    assert_allclose(params[4:], [0, 0, 0], atol=1e-11)

# TEST: deliberately fail to converge

