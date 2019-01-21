from __future__ import division

import numpy as np
from numpy.testing import assert_allclose
import pytest

import radical.constants as constants
from radical.coordinates import radec_to_lm
import radical.phaserotate as phaserotate


def test_phaserotatate_onto_current_phasecenter(mockms):
    # Rotate to current phase center
    mockms.data[:, :, :] = 1
    uvw, rotated = phaserotate.phase_rotate(mockms.uvw, mockms.data, mockms.ra0, mockms.dec0, mockms.ra0, mockms.dec0, mockms.lambdas)

    assert_allclose(rotated, mockms.data)
    assert_allclose(uvw, mockms.uvw)


@pytest.mark.parametrize(
    'ra, dec',
    [
        (0.6, 0.1),
        (0.2, -0.1),
        (-0.3, 0.3),
    ]
)
def test_phaserotate_and_back_again(mockms, ra, dec):
    # Rotate away
    uvw, data = phaserotate.phase_rotate(mockms.uvw, mockms.data, ra, dec, mockms.ra0, mockms.dec0, mockms.lambdas)
    # ...and back again
    uvw, data = phaserotate.phase_rotate(uvw, data, mockms.ra0, mockms.dec0, ra, dec, mockms.lambdas)

    assert_allclose(uvw, mockms.uvw)
    assert_allclose(data, mockms.data)


@pytest.mark.parametrize(
    'ra, dec',
    [
        (0.6, 0.1),
        (0.2, -0.1),
        (-0.3, 0.3),
    ]
)
def test_phaserotate_to_source(mockms, ra, dec):
    u, v, w = mockms.u_lambda, mockms.v_lambda, mockms.w_lambda
    l, m = radec_to_lm(ra, dec, mockms.ra0, mockms.dec0)

    mockms.data = np.array([1, 0, 0, 1]) * np.exp(
        2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1))
    )[:, :, None]

    uvw, data = phaserotate.phase_rotate(mockms.uvw, mockms.data, ra, dec, mockms.ra0, mockms.dec0, mockms.lambdas)

    # Assert baseline lengths are unchanged
    assert_allclose((uvw**2).sum(axis=1), (mockms.uvw**2).sum(axis=1))

    # Assert rotated data now all has phase 0
    desired = np.ones_like(data[:, :, [True, False, False, True]])
    assert_allclose(data[:, :, [True, False, False, True]], desired)
