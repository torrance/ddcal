from __future__ import division

from astropy.coordinates import SkyCoord
import astropy.units as unit
import numpy as np
from numpy.testing import assert_allclose
import pytest

from radical.calibrate import solve
from radical.coordinates import radec_to_lm
import radical.skymodel as skymodel
from radical.solution import Solution


def test_simple_firstorder(mockms, mockcomp1, mockcomp2):
    # Create skymodel
    coord1 = SkyCoord('0:0:0 0:0:0', unit=(unit.hourangle, unit.degree))
    coord2 = SkyCoord('0:0:0 0:0:30', unit=(unit.hourangle, unit.degree))

    mockcomp1.ra, mockcomp1.dec = coord1.ra.radian, coord1.dec.radian
    mockcomp2.ra, mockcomp2.dec = coord2.ra.radian, coord2.dec.radian

    src = skymodel.Model('mymodel', [mockcomp1, mockcomp2])

    # Create data
    u, v, w = mockms.uvw.T[:, :, None] / mockms.lambdas
    mockms.data *= 0

    l, m = radec_to_lm(mockcomp1.ra, mockcomp1.dec, mockms.ra0, mockms.dec0)
    mockms.data[:, :, 0] += 5 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))
    mockms.data[:, :, 3] += 4 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))

    l, m = radec_to_lm(mockcomp2.ra, mockcomp2.dec, mockms.ra0, mockms.dec0)
    mockms.data[:, :, 0] += 3 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))
    mockms.data[:, :, 3] += 2 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))

    solution = Solution(ncomp=2)
    solve(src, solution, mockms, 1)

    assert_allclose([5, 4, 3, 2], solution.get_params(0), atol=1e-1)
    assert_allclose([0, 0], solution.get_params(1)[-2:], atol=5e-6)


def test_firstorder(mockms, mockcomp1, mockcomp2):
    # Create skymodel
    coord1 = SkyCoord('0:0:0 0:0:0', unit=(unit.hourangle, unit.degree))
    coord2 = SkyCoord('0:0:0 0:0:30', unit=(unit.hourangle, unit.degree))

    mockcomp1.ra, mockcomp1.dec = coord1.ra.radian, coord1.dec.radian
    mockcomp2.ra, mockcomp2.dec = coord2.ra.radian, coord2.dec.radian

    src = skymodel.Model('mymodel', [mockcomp1, mockcomp2])

    # Create data
    u, v, w = mockms.uvw.T[:, :, None] / mockms.lambdas
    mockms.data *= 0

    l, m = radec_to_lm(mockcomp1.ra, mockcomp1.dec, mockms.ra0, mockms.dec0)
    mockms.data[:, :, 0] += 5 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))
    mockms.data[:, :, 3] += 4 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))

    l, m = radec_to_lm(mockcomp2.ra, mockcomp2.dec, mockms.ra0, mockms.dec0)
    mockms.data[:, :, 0] += 3 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))
    mockms.data[:, :, 3] += 2 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))

    x, y = 1e-5, -3e-5
    U, V = mockms.U, mockms.V
    phases = x * U + y * V
    phases = phases[mockms.ant1] - phases[mockms.ant2]

    mockms.data *= np.exp(-1j * phases)[:, None, None]

    solution = Solution(ncomp=2)
    solve(src, solution, mockms, 1)

    assert_allclose([5, 4, 3, 2], solution.get_params(0), atol=1e-1)
    assert_allclose([x, y], solution.get_params(1)[-2:], atol=5e-6)


def test_firstorder_with_noise(mockms, mockcomp1, mockcomp2):
    # Create skymodel
    coord1 = SkyCoord('0:0:0 0:0:0', unit=(unit.hourangle, unit.degree))
    coord2 = SkyCoord('0:0:0 0:0:30', unit=(unit.hourangle, unit.degree))

    mockcomp1.ra, mockcomp1.dec = coord1.ra.radian, coord1.dec.radian
    mockcomp2.ra, mockcomp2.dec = coord2.ra.radian, coord2.dec.radian

    src = skymodel.Model('mymodel', [mockcomp1, mockcomp2])

    # Create data
    u, v, w = mockms.uvw.T[:, :, None] / mockms.lambdas
    mockms.data *= 0

    l, m = radec_to_lm(mockcomp1.ra, mockcomp1.dec, mockms.ra0, mockms.dec0)
    mockms.data[:, :, 0] += 5 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))
    mockms.data[:, :, 3] += 4 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))

    l, m = radec_to_lm(mockcomp2.ra, mockcomp2.dec, mockms.ra0, mockms.dec0)
    mockms.data[:, :, 0] += 3 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))
    mockms.data[:, :, 3] += 2 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))

    x, y = 1e-5, -3e-5
    U, V = mockms.U, mockms.V
    antphases = x * U + y * V
    phases = antphases[mockms.ant1] - antphases[mockms.ant2]

    mockms.data *= np.exp(-1j * phases)[:, None, None]
    mockms.data += np.random.normal(0, 25, mockms.data.shape) + 1j * np.random.normal(0, 25, mockms.data.shape)

    solution = Solution(ncomp=2)
    solve(src, solution, mockms, 1)

    assert_allclose(solution.phases(mockms.U, mockms.V), antphases, atol=5e-2)


def test_firstorder_modelerror(mockms, mockcomp1, mockcomp2):
    # Create skymodel
    coord1 = SkyCoord('0:0:0 0:0:0', unit=(unit.hourangle, unit.degree))
    coord2 = SkyCoord('0:0:0 0:0:30', unit=(unit.hourangle, unit.degree))
    coord3 = SkyCoord('0:0:0 0:0:35', unit=(unit.hourangle, unit.degree))  # True position

    mockcomp1.ra, mockcomp1.dec = coord1.ra.radian, coord1.dec.radian
    mockcomp2.ra, mockcomp2.dec = coord2.ra.radian, coord2.dec.radian

    src = skymodel.Model('mymodel', [mockcomp1, mockcomp2])

    # Create data
    u, v, w = mockms.uvw.T[:, :, None] / mockms.lambdas
    mockms.data *= 0

    l, m = radec_to_lm(mockcomp1.ra, mockcomp1.dec, mockms.ra0, mockms.dec0)
    mockms.data[:, :, 0] += 5 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))
    mockms.data[:, :, 3] += 4 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))

    l, m = radec_to_lm(coord3.ra.radian, coord3.dec.radian, mockms.ra0, mockms.dec0)
    mockms.data[:, :, 0] += 3 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))
    mockms.data[:, :, 3] += 2 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))

    x, y = 1e-5, -3e-5
    U, V = mockms.U, mockms.V
    antphases = x * U + y * V
    phases = antphases[mockms.ant1] - antphases[mockms.ant2]

    mockms.data *= np.exp(-1j * phases)[:, None, None]

    solution = Solution(ncomp=2)
    solve(src, solution, mockms, 1)

    assert_allclose(solution.phases(mockms.U, mockms.V), antphases, atol=1e-1)


def test_firstorder_modelerror_with_noise(mockms, mockcomp1, mockcomp2):
    # Create skymodel
    coord1 = SkyCoord('0:0:0 0:0:0', unit=(unit.hourangle, unit.degree))
    coord2 = SkyCoord('0:0:0 0:0:30', unit=(unit.hourangle, unit.degree))
    coord3 = SkyCoord('0:0:0 0:0:35', unit=(unit.hourangle, unit.degree))  # True position

    mockcomp1.ra, mockcomp1.dec = coord1.ra.radian, coord1.dec.radian
    mockcomp2.ra, mockcomp2.dec = coord2.ra.radian, coord2.dec.radian

    src = skymodel.Model('mymodel', [mockcomp1, mockcomp2])

    # Create data
    u, v, w = mockms.uvw.T[:, :, None] / mockms.lambdas
    mockms.data *= 0

    l, m = radec_to_lm(mockcomp1.ra, mockcomp1.dec, mockms.ra0, mockms.dec0)
    mockms.data[:, :, 0] += 5 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))
    mockms.data[:, :, 3] += 4 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))

    l, m = radec_to_lm(coord3.ra.radian, coord3.dec.radian, mockms.ra0, mockms.dec0)
    mockms.data[:, :, 0] += 3 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))
    mockms.data[:, :, 3] += 2 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))

    x, y = 1e-5, -3e-5
    U, V = mockms.U, mockms.V
    antphases = x * U + y * V
    phases = antphases[mockms.ant1] - antphases[mockms.ant2]

    mockms.data *= np.exp(-1j * phases)[:, None, None]
    mockms.data += np.random.normal(0, 25, mockms.data.shape) + 1j * np.random.normal(0, 25, mockms.data.shape)

    solution = Solution(ncomp=2)
    solve(src, solution, mockms, 1)

    assert_allclose(solution.phases(mockms.U, mockms.V), antphases, atol=1e-1)


def test_secondorder_modelerror(mockms, mockcomp1, mockcomp2):
    # Create skymodel
    coord1 = SkyCoord('0:0:0 0:0:0', unit=(unit.hourangle, unit.degree))
    coord2 = SkyCoord('0:0:0 0:0:30', unit=(unit.hourangle, unit.degree))
    coord3 = SkyCoord('0:0:0 0:0:35', unit=(unit.hourangle, unit.degree))  # True position

    mockcomp1.ra, mockcomp1.dec = coord1.ra.radian, coord1.dec.radian
    mockcomp2.ra, mockcomp2.dec = coord2.ra.radian, coord2.dec.radian

    src = skymodel.Model('mymodel', [mockcomp1, mockcomp2])

    # Create data
    u, v, w = mockms.uvw.T[:, :, None] / mockms.lambdas
    mockms.data *= 0

    l, m = radec_to_lm(mockcomp1.ra, mockcomp1.dec, mockms.ra0, mockms.dec0)
    mockms.data[:, :, 0] += 5 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))
    mockms.data[:, :, 3] += 4 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))

    l, m = radec_to_lm(coord3.ra.radian, coord3.dec.radian, mockms.ra0, mockms.dec0)
    mockms.data[:, :, 0] += 3 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))
    mockms.data[:, :, 3] += 2 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))

    x, y, xx, xy, yy = 1e-5, -3e-5, 1e-7, -2.5e-8, 3e-8
    U, V = mockms.U, mockms.V
    antphases = x * U + y * V + xx * U**2 + xy * U * V + yy * V**2
    phases = antphases[mockms.ant1] - antphases[mockms.ant2]

    mockms.data *= np.exp(-1j * phases)[:, None, None]

    solution = Solution(ncomp=2)
    solve(src, solution, mockms, 1)
    solve(src, solution, mockms, 2)

    assert_allclose(solution.phases(mockms.U, mockms.V), antphases, atol=1e-1)


def test_secondorder_modelerror_with_noise(mockms, mockcomp1, mockcomp2):
    # Create skymodel
    coord1 = SkyCoord('0:0:0 0:0:0', unit=(unit.hourangle, unit.degree))
    coord2 = SkyCoord('0:0:0 0:0:30', unit=(unit.hourangle, unit.degree))
    coord3 = SkyCoord('0:0:0 0:0:35', unit=(unit.hourangle, unit.degree))  # True position

    mockcomp1.ra, mockcomp1.dec = coord1.ra.radian, coord1.dec.radian
    mockcomp2.ra, mockcomp2.dec = coord2.ra.radian, coord2.dec.radian

    src = skymodel.Model('mymodel', [mockcomp1, mockcomp2])

    # Create data
    u, v, w = mockms.uvw.T[:, :, None] / mockms.lambdas
    mockms.data *= 0

    l, m = radec_to_lm(mockcomp1.ra, mockcomp1.dec, mockms.ra0, mockms.dec0)
    mockms.data[:, :, 0] += 5 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))
    mockms.data[:, :, 3] += 4 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))

    l, m = radec_to_lm(coord3.ra.radian, coord3.dec.radian, mockms.ra0, mockms.dec0)
    mockms.data[:, :, 0] += 3 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))
    mockms.data[:, :, 3] += 2 * np.exp(2j * np.pi * (u*l + v*m + w*(np.sqrt(1 - l**2 - m**2) - 1)))

    x, y, xx, xy, yy = 1e-5, -3e-5, 1e-7, -2.5e-8, 3e-8
    U, V = mockms.U, mockms.V
    antphases = x * U + y * V + xx * U**2 + xy * U * V + yy * V**2
    phases = antphases[mockms.ant1] - antphases[mockms.ant2]

    mockms.data *= np.exp(-1j * phases)[:, None, None]
    mockms.data += np.random.normal(0, 25, mockms.data.shape) + 1j * np.random.normal(0, 25, mockms.data.shape)

    solution = Solution(ncomp=2)
    solve(src, solution, mockms, 1)
    solve(src, solution, mockms, 2)

    assert_allclose(solution.phases(mockms.U, mockms.V), antphases, atol=1e-1)
