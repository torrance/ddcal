from __future__ import print_function, division

from astropy.coordinates import AltAz, SkyCoord
import astropy.units as u
import numpy as np


def radec_to_lm(ra, dec, ra0, dec0):
    deltaAlpha = ra - ra0
    sinDeltaAlpha = np.sin(deltaAlpha)
    cosDeltaAlpha = np.cos(deltaAlpha)
    sinDec = np.sin(dec)
    cosDec = np.cos(dec)
    sinDec0 = np.sin(dec0)
    cosDec0 = np.cos(dec0)

    l = cosDec * sinDeltaAlpha
    m = sinDec * cosDec0 - cosDec * sinDec0 * cosDeltaAlpha
    return np.array([l, m])


def lm_to_radec(l, m, ra0, dec0):
    n = np.sqrt(1 - l**2 - m**2)
    cosDec0 = np.cos(dec0)
    sinDec0 = np.sin(dec0)
    deltaAlpha = np.arctan2(l, n * cosDec0 - m * sinDec0)

    ra = deltaAlpha + ra0
    dec = np.arcsin(m * cosDec0 + n * sinDec0)
    return np.array([ra, dec])


def radec_to_altaz(ra, dec, time, pos):
    coord = SkyCoord(ra, dec, unit=(u.radian, u.radian))
    coord.time = time + pos.lon.hourangle
    coord = coord.transform_to(AltAz(obstime=time, location=pos))
    return coord.alt.rad, coord.az.rad
