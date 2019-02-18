from __future__ import division

import logging
import sys

from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as units
from astropy.wcs import WCS
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from radical.coordinates import radec_to_lm, lm_to_radec


logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')


def make(sources, solutions, mset, radius, scale, center, oversample, smoothing_kernel):
    tec = provision(mset, radius, scale, center)
    interpolate(tec, sources, solutions, mset, oversample, smoothing_kernel)
    return tec


def provision(mset, radius, scale, center):
    width, height = int((2 * radius) // scale), int((2 * radius) // scale)
    center_x, center_y = int(width // 2), int(height // 2)
    logger.info("Creating TEC image of dimensions (%d, %d)", width, height)

    data = np.zeros((1, 1, len(mset.antids), height, width), dtype=np.float)  # [time, frequency, antennas, dec, ra]
    tec = fits.PrimaryHDU(data)

    tec.header['CTYPE1'] = 'RA---SIN'
    tec.header['CRPIX1'] = center_x
    tec.header['CRVAL1'] = center.ra.deg
    tec.header['CDELT1'] = scale
    tec.header['CUNIT1'] = 'deg'

    tec.header['CTYPE2'] = 'DEC--SIN'
    tec.header['CRPIX2'] = center_y
    tec.header['CRVAl2'] = center.dec.deg
    tec.header['CDELT2'] = scale
    tec.header['CUNIT2'] = 'deg'

    tec.header['CTYPE3'] = 'ANTENNA'
    tec.header['CRPIX3'] = 1
    tec.header['CRVAL3'] = 0

    tec.header['CTYPE4'] = 'FREQ'
    tec.header['CRPIX4'] = 1
    tec.header['CRVAL4'] = mset.midfreq[0]
    tec.header['CDELT4'] = 1
    tec.header['CUNIT4'] = 'Hz'

    tec.header['CTYPE5'] = 'TIME'
    tec.header['CRPIX5'] = 1
    tec.header['CRVAL5'] = mset.midtime  # FIXME
    tec.header['CDELT5'] = 1

    tec.header['HISTORY'] = 'radical ' + ' '.join(sys.argv[1:])

    return tec


def interpolate(tec, sources, solutions, mset, oversample=1, smoothing_kernel=0):
    # Extract dimensions and world coordinates
    data = tec.data
    height, width = data.shape[3:]
    center = SkyCoord(tec.header['CRVAL1'], tec.header['CRVAL2'], unit=(units.degree, units.degree))
    wcs = WCS(tec.header)

    # Create lists of lm coordinates in the FITS projection for calibration directions
    ras = np.array([src.ra for src in sources])
    decs = np.array([src.dec for src in sources])
    directions_lm = radec_to_lm(ras, decs, center.ra.rad, center.dec.rad)

    # Solve phases for each antenna for each calibration direction
    phases = np.empty((len(mset.antids), len(sources)))
    for i, solution in enumerate(solutions):
        phases[:, i] = solution.phases(mset.U, mset.V)

    # Get oversampled l,m values for TEC file
    xx, yy = np.meshgrid(range(0, oversample * width), range(0, oversample * height))
    pixels = np.array([xx.flatten(), yy.flatten()]).T

    ret = wcs.all_pix2world([[x / oversample - 1/oversample, y / oversample - 1/oversample, 0, 0, 0] for x, y in pixels], 0)
    grid_lm = radec_to_lm(np.radians(ret.T[0]), np.radians(ret.T[1]), center.ra.rad, center.dec.rad)

    from scipy.interpolate import Rbf
    for i in mset.antids:
        # Compute interpolated phases
        phases_grid = griddata(directions_lm.T, phases[i], grid_lm.T, method='nearest', fill_value=0)
        # phases_grid = Rbf(directions_lm[0], directions_lm[1], phases[i], smooth=0.1)(grid_lm[0], grid_lm[1])
        # phases_grid = nearestneighbour(directions_lm[0], directions_lm[1], phases[i], grid_lm.T, maxradius=3.0)
        phases_grid = np.reshape(phases_grid, (oversample * height, oversample * width))  # [ dec, ra ]

        # Gaussian smooth
        phases_grid = gaussian_filter(phases_grid, oversample * smoothing_kernel, mode='constant', cval=0)

        # Downsample
        phases_grid = phases_grid[oversample//2::oversample, oversample//2::oversample]

        data[0, 0, i] = phases_grid / 8.44797245E9 * mset.midfreq


def nearestneighbour(xs, ys, zs, points, maxradius=1):
    maxradiussquared = np.arcsin(np.radians(maxradius))**2  #  Conversion to lm space, only approx

    distsquared = (points[:, 0, None] - xs[None, :])**2 + (points[:, 1, None] - ys[None, :])**2  # [points, directions]

    idx_closest = np.argmin(distsquared, axis=1)  # [points]
    distsquared = np.min(distsquared, axis=1)

    closest = zs[idx_closest]
    closest[distsquared > maxradiussquared] = 0

    return closest
