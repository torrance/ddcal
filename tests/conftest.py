import os
import numpy as np
import pytest

import radical.constants as constants
from radical.measurementset import MeasurementSet


@pytest.fixture()
def mockcomp():
    return MockComponent()


@pytest.fixture()
def mockms():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    mset = MeasurementSet(dir_path + '/data/dummy.ms', refant=0)

    # Add additional frequency data
    freqs = np.load(dir_path + '/data/freqs.npy')
    mset.freqs = freqs
    mset.midfreq = np.array([(min(freqs) + max(freqs)) / 2])
    mset.lambdas = constants.c / freqs
    mset.midlambda = constants.c / mset.midfreq

    # Update u,v,w_lambda and resize data
    mset.u_lambda, mset.v_lambda, mset.w_lambda = mset.uvw.T[:, :, None] / mset.lambdas
    mset.data = np.empty((len(mset.uvw), len(mset.freqs), 4), dtype=np.complex128)

    return mset


class MockComponent(object):
    def __init__(self):
        self.position = self

    def to_string(self, fmt):
        return "%g %g" % (self.ra, self.dec)


