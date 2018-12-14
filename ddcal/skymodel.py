from __future__ import print_function, division

import sys

from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np


def parse(f):
    models = []

    next(f)  # Skip version info
    for line in f:
        parts = line.split()
        if parts[0] == 'source':
            models.append(source_parser(f))
        elif parts[0] == '#':
            pass
        else:
            raise SkyModelParseError("Unexpected line: %s" % line)

    return models


def source_parser(f):
    name, components = None, []

    for line in f:
        parts = line.split()
        if parts[0] == 'name':
            name = ' '.join(parts[1:]).strip('"')
        elif parts[0] == 'component':
            components.append(component_parser(f))
        elif parts[0] == '}':
            if name and components:
                return Model(name, components)
            else:
                raise SkyModelParseError("Unexpected }")
        else:
            raise SkyModelParseError("Skymodel parsing error: %s" % line)

    raise SkyModelParseError("Unexpected EOF")


def component_parser(f):
    position, measurements = None, []

    for line in f:
        parts = line.split()
        if parts[0] == 'position':
            position = SkyCoord(parts[1], parts[2], unit=(u.hourangle, u.degree))
        elif parts[0] == 'type':
            pass
        elif parts[0] == 'shape':
            pass
        elif parts[0] == 'measurement':
            measurements.append(measurement_parser(f))
        elif parts[0] == '}':
            if position and measurements:
                measurements = np.sort(np.array(measurements), axis=0)
                return Component(position, measurements)
            else:
                raise SkyModelParseError("Unexpected }")
        else:
            raise SkyModelParseError("Skymodel parsing error: %s" % line)

    raise SkyModelParseError("Unexpected EOF")


def measurement_parser(f):
    frequency, fluxdensity = None, []

    for line in f:
        parts = line.split()
        if parts[0] == 'frequency':
            # Assume MHz for now
            frequency = float(parts[1]) * 1E6
        elif parts[0] == 'fluxdensity':
            fluxdensity = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
        elif parts[0] == '}':
            if frequency and fluxdensity:
                return [frequency] + fluxdensity
            else:
                raise SkyModelParseError("Unexpected {")
        else:
            raise SkyModelParseError("SkyModel parsing error: %s" % line)

    raise SkyModelParseError("Unexpected EOF")


class Model(object):
    def __init__(self, name, components):
        self.name = name
        self.components = components


class Component(object):
    def __init__(self, position, measurements):
        self.position = position
        self.measurements = measurements

        logfreq = np.log(self.measurements.T[0])
        logflux = np.log(self.measurements.T[1])
        self.coeffs = np.polyfit(logfreq, logflux, min(len(logfreq) - 1, 3))

    @property
    def ra(self):
        return self.position.ra.rad

    @property
    def dec(self):
        return self.position.dec.rad

    def flux(self, frequency):
        logfreq = np.log(frequency)
        logflux = 0

        for i, c in enumerate(reversed(self.coeffs)):
           logflux += c * logfreq**i

        return np.exp(logflux)

    def apparent(self, frequency):
        # TODO
        return self.flux(frequency)


class SkyModelParseError(Exception):
    pass

