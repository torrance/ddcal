from __future__ import division

import numpy as np
from numpy.testing import assert_allclose
import pytest

from radical.coordinates import radec_to_lm, lm_to_radec

@pytest.mark.parametrize(
    'ra, dec',
    [
        (-30, -72),
        (90, -45),
        (90, 45),
    ]
)
def test_there_and_back_again(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    ra0, dec0 = np.radians(90), np.radians(-45)

    l, m = radec_to_lm(ra, dec, ra0, dec0)
    print(l, m)
    ra_back, dec_back = lm_to_radec(l, m, ra0, dec0)


    assert_allclose([ra_back, dec_back], [ra, dec])


@pytest.mark.parametrize(
    'ra, dec, expected_l, expected_m',
    [
        (90, 90, 0, 0),
        (0, 45, 0, -np.sqrt(1/2)),
        (270, 0, -1, 0),
    ]
)
def test_radec_to_lm(ra, dec, expected_l, expected_m):
    ra, dec = np.radians(ra), np.radians(dec)
    ra0, dec0 = np.radians(0), np.radians(90)

    l, m = radec_to_lm(ra, dec, ra0, dec0)

    assert_allclose([l, m], [expected_l, expected_m], atol=1e-12)


