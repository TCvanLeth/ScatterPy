#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ScatterPy T-matrix simulation of electromagnetic scattering by nonspherical
particles.

Copyright (C) 2019 Thomas C. van Leth
-------------------

This module contains special mathematical functions used in the T-matrix
calculations.
"""

import numba as nb
import numpy as np
from scipy import special


# special mathematical functions
def sph_jn(n, z):
    """
    Spherical j Bessel function of order n.
    """
    return special.jv(n + 1 / 2, z) * np.sqrt(np.pi / 2) / (np.sqrt(z))


def sph_yn(n, z):
    """
    Spherical y Bessel function of order n.
    """
    return special.yv(n + 1 / 2, z) * np.sqrt(np.pi / 2) / (np.sqrt(z))


#@nb.jit(nb.types.UniTuple(nb.float64[:,:,:], 2)(nb.float64[:], nb.int64, nb.optional(nb.int64)), nopython=True)
def wigner_d(x, n_max, m_max=None):
    """
    Create wigner d functions.

    x --> sample points
    n_max
    m_max
    """
    if m_max is None:
        m_max = n_max

    dv0 = np.zeros(x.shape + (m_max + 1, n_max))
    dv1 = dv0.copy()
    dv2 = dv0.copy()
    for m in range(m_max + 1):
        qs = np.sqrt(1 - x**2)
        if m == 0:
            d1 = np.ones_like(x)
            d2 = x
            m1 = 0
        else:
            d1 = np.zeros_like(x)
            count = np.arange(0, 2 * m, 2)
            d2 = np.prod(np.sqrt((count + 1) / (count + 2))) * qs**m
            m1 = m-1

        qs = np.where(qs == 0, np.array([1.]), qs)  # deal with singularity
        for n in range(m1, n_max):
            n1 = n + 1
            nm0 = np.sqrt(n1**2 - m**2)
            nm1 = np.sqrt((n1 + 1)**2 - m**2)

            d3 = ((2 * n1 + 1) * x * d2 - nm0 * d1)
            dv2[..., m, n] = (n1 * d3 - (n1+1) * nm0 * d1) / ((2*n1 + 1) * qs)
            dv1[..., m, n] = d2
            d1 = d2
            d2 = d3 / nm1
    return dv1, dv2
