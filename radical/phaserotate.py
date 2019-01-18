from __future__ import print_function, division

import logging
import time as tm

from casacore.measures import measures
from casacore.quanta import quantity
from casacore.tables import taql
from numba import njit, float64, complex64, complex128, prange
import numpy as np

import radical.constants as constants


logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')


def phase_rotate(uvw, data, ra, dec, metadata):
    dm = metadata.dm
    phasecenter = dm.direction(
        'j2000',
        quantity(ra, 'rad'),
        quantity(dec, 'rad'),
    )
    dm.do_frame(phasecenter)

    times = metadata.times
    ant1, ant2 = metadata.ant1, metadata.ant2
    antennas = metadata.antennas
    lambdas = metadata.lambdas

    # Recalculate uvw for new phase position
    new_uvw = np.empty_like(uvw)

    # Process visibilities by time so that we calculate antenna baselines
    # just once
    start = tm.time()
    for time in set(times):
        epoch = dm.epoch('UTC', quantity(time, 's'))
        dm.do_frame(epoch)
        baselines = dm.as_baseline(antennas)
        antenna_uvw = np.reshape(dm.to_uvw(baselines)['xyz'].get_value(), (-1, 3))

        # Select only those rows for the current time
        # and update uvw values
        idx = times == time
        new_uvw[idx] = antenna_uvw[ant1[idx]] - antenna_uvw[ant2[idx]]
    elapsed = tm.time() - start
    logger.debug("Phase rotated uvw elapsed: %g", elapsed)

    # Calculate phase offset
    start = tm.time()
    new_data = woffset(data, uvw.T[2], new_uvw.T[2], lambdas)
    elapsed = tm.time() - start
    logger.debug("Phase rotated visibilities elapsed: %g", elapsed)

    return new_uvw, new_data


@njit([complex128[:, :, :](complex128[:, :, :], float64[:], float64[:], float64[:])], parallel=True)
def woffset(data, oldw, neww, lambdas):
    offset = -2j * np.pi * (neww - oldw)
    phase = np.empty_like(data)
    for row in prange(0, phase.shape[0]):
        tmp = offset[row] / lambdas
        for pol in range(0, data.shape[2]):
            phase[row, :, pol] = tmp

    return data * np.exp(phase)


class Metadata(object):
    def __init__(self, mset):
        self.dm = measures()

        # Find observatory position
        obsnames = mset.OBSERVATION.getcol('TELESCOPE_NAME')
        if len(obsnames) == 1:
            obspos = self.dm.observatory(obsnames[0])
            self.dm.do_frame(obspos)
        else:
            raise Exception("Failed to work out the telescope name of this observation")

        # Get antenna positions
        antennas = mset.ANTENNA.getcol('POSITION')
        self.antennas = self.dm.position(
            'itrf',
            quantity(antennas.T[0], 'm'),
            quantity(antennas.T[1], 'm'),
            quantity(antennas.T[2], 'm'),
        )

        # Get lambdas
        self.lambdas = constants.c / mset.SPECTRAL_WINDOW.getcell("CHAN_FREQ", 0)

        # Load times and antenna IDs
        self.times = mset.getcol("TIME")
        self.ant1 = mset.getcol("ANTENNA1")
        self.ant2 = mset.getcol("ANTENNA2")

