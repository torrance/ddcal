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
    metadata = phaserotate.Metadata(mockms)
    u, v, w = mockms.UVW.T[:, :, None] / mockms.CHAN_FREQ

    l, m = radec_to_lm(mockms.RA0, mockms.DEC0, mockms.RA0, mockms.DEC0)
    data = np.array([Ax, 0, 0, Ay]) * np.exp(
        2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1))
    )[:, :, None]

    solution = Solution()

    mockcomp.ra = mockms.RA0
    mockcomp.dec = mockms.DEC0

    calibrate.solve(mockcomp, solution, mockms.UVW, data, mockms.U, mockms.V, mockms.ANTENNA1, mockms.ANTENNA2, metadata, 1)
    params = solution.get_params(2)

    assert_allclose(params, [Ax, Ay, 0, 0, 0, 0, 0], atol=1e-15)



@pytest.mark.parametrize(
    'Ax, Ay, ra, dec',
    [
        (4, 3.5, 0.19, -0.32),
    ]
)
def test_firstorder_amplitudefit(mockms, mockcomp, Ax, Ay, ra, dec):
    metadata = phaserotate.Metadata(mockms)
    u, v, w = mockms.UVW.T[:, :, None] / (constants.c / mockms.CHAN_FREQ)

    l, m = radec_to_lm(ra, dec, mockms.RA0, mockms.DEC0)
    data = np.array([Ax, 0, 0, Ay]) * np.exp(
        2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1))
    )[:, :, None]

    solution = Solution()

    mockcomp.ra = ra
    mockcomp.dec = dec

    calibrate.solve(mockcomp, solution, mockms.UVW, data, mockms.U, mockms.V, mockms.ANTENNA1, mockms.ANTENNA2, metadata, 1)
    params = solution.get_params(2)

    assert_allclose(params[:2], [Ax, Ay], rtol=1e-2)
    assert_allclose(params[2:4], [0, 0], atol=1e-7)
    assert_allclose(params[4:], [0, 0, 0], atol=1e-11)


# TEST: deliberately fail to converge

