import os
import numpy as np
import pytest


@pytest.fixture()
def mockcomp():
    return MockComponent()


@pytest.fixture(scope='module')
def mockms():
    return MockMS()


class MockComponent(object):
    def __init__(self):
        self.position = self

    def to_string(self, fmt):
        return "%g %g" % (self.ra, self.dec)


class MockMS(object):
    def __init__(self):
        self.ANTENNA = self
        self.OBSERVATION = self
        self.SPECTRAL_WINDOW = self

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.ANTENNA1 = np.load(dir_path + '/data/ant1.npy')
        self.ANTENNA2 = np.load(dir_path + '/data/ant2.npy')
        self.TIME = np.load(dir_path + '/data/times.npy')
        self.UVW = np.load(dir_path + '/data//uvw.npy')
        self.U = np.load(dir_path + '/data/U.npy')
        self.V = np.load(dir_path + '/data/V.npy')
        self.POSITION = np.load(dir_path + '/data/antennas.npy')
        self.CHAN_FREQ = np.load(dir_path + '/data/freqs.npy')
        self.TELESCOPE_NAME = ['MWA']
        self.RA0 = 0.09
        self.DEC0 = -0.47

    def getcol(self, name):
        return getattr(self, name)

    def getcell(self, name, index):
        return getattr(self, name)
