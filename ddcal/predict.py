from __future__ import print_function, division

import numpy as np

from .coordinates import radec_to_lm


def source(comp, ra0, dec0, u, v, w, freqs):
    """
    TODO: How to handle multi-component sources??
    u, v, w: [correlations]
    """
    l, m = radec_to_lm(comp.ra, comp.dec, ra0, dec0)
    model = comp.apparent(freqs)[None, :] / np.sqrt(1 - l**2 - m**2) * np.exp(
        2j * np.pi * (u * l + v * m + w * (np.sqrt(1 - l**2 - m**2) - 1))
    )

    return model






