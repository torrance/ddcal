from __future__ import division

from casacore.tables import table, taql
import numpy as np

import radical.constants as constants


class MeasurementSet(object):
    def __init__(self, filename, refant=0, datacolumn='CORRECTED_DATA'):
        self.mset = table(filename, readonly=False, ack=False)
        mset = self.mset

        self.antids = np.array(range(0, len(mset.ANTENNA)))
        self.ra0, self.dec0 = mset.FIELD.getcell('PHASE_DIR', 0)[0]

        self.freqs = mset.SPECTRAL_WINDOW.getcell('CHAN_FREQ', 0)
        self.midfreq = np.array([(min(self.freqs) + max(self.freqs)) / 2])
        self.lambdas = constants.c / self.freqs
        self.midlambda = constants.c / self.midfreq

        # Calculate antenna positions wrt refant antenna
        times = sorted(set(mset.getcol('TIME')))
        self.midtime = times[len(times) // 2]
        midtime = self.midtime
        tmp = taql("select UVW, ANTENNA2 from $mset where TIME = $midtime and ANTENNA1 = $refant")
        (_U, _V, _), antennas = tmp.getcol('UVW').T, tmp.getcol('ANTENNA2')

        # Force U, V indices to align with antenna IDs
        self.U = np.zeros_like(self.antids, dtype=np.float64)
        self.U[antennas] = _U
        self.V = np.zeros_like(self.antids, dtype=np.float64)
        self.V[antennas] = _V

        # Load data and associated row information
        # Filter out flagged rows, and autocorrelations
        self.filtered = taql("select * from $mset where not FLAG_ROW and ANTENNA1 <> ANTENNA2")
        self.ant1 = self.filtered.getcol('ANTENNA1')
        self.ant2 = self.filtered.getcol('ANTENNA2')
        self.uvw = self.filtered.getcol('UVW')
        self.u_lambda, self.v_lambda, self.w_lambda = self.uvw.T[:, :, None] / self.lambdas
        self._data = None
        self.datacolumn = datacolumn

    @property
    def data(self):
        if self._data is None:
            flags = self.filtered.getcol('FLAG')
            self._data = np.complex128(self.filtered.getcol(self.datacolumn))
            self._data[flags] = np.nan

        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    def __getattr__(self, name):
        return getattr(self.mset, name)
