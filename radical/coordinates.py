from __future__ import print_function, division

from astropy.coordinates import AltAz, SkyCoord
import astropy.units as u
import numpy as np


def radec_to_lm(ra, dec, ra0, dec0):
    l = np.cos(dec) * np.sin(ra - ra0)
    m = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(ra - ra0)
    return np.array([l, m])


def lm_to_radec(l, m, ra0, dec0):
    n = np.sqrt(1 - l**2 - m**2)
    delta_ra = np.arctan2(l, n * np.cos(dec0) - m * np.sin(dec0))
    ra = ra0 + delta_ra
    dec = np.arcsin(m * np.cos(dec0) + n * np.sin(dec0))
    return np.array([ra, dec])


def radec_to_altaz(ra, dec, time, pos):
    coord = SkyCoord(ra, dec, unit=(u.radian, u.radian))
    coord.time = time + pos.lon.hourangle
    coord = coord.transform_to(AltAz(obstime=time, location=pos))
    return coord.alt.rad, coord.az.rad
