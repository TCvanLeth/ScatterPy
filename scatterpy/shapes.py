#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ScatterPy T-matrix simulation of electromagnetic scattering by nonspherical
particles.

Copyright (C) 2019 Thomas C. van Leth
-------------------

Particle shapes
===============

This module provides a collection of shape functions for rotationally
symmetric particles, and for raindrop shapes in particular.

General shapes
--------------
.. autosummary::
    spheroid
    chebyshev
    gen_chebyshev

Raindrop shapes
---------------
.. autosummary::
    dropshape_CB90
    dropshape_Th07
    dropshape_An99
"""

import numpy as np
from scipy.interpolate import interp1d


###############################################################################
# The following are generic rotationally symmetric particle shapes in spherical
# coordinates.

def spheroid(c):
    """
    Spheroid radius as a function of theta angle.

    Parameters
    ----------
    c : float or ndarray
        The ratio of horizontal to rotational axes.

    Returns
    -------
    func(x) : function
        A function that returns the radius and first derivative of the
        radius at the theta angles determined by Gaussian quadrature x.
    """
    c = c[..., None]
    def func(x):
        cc = c**2
        ss = 1 - x**2
        rr = 1 / (ss + cc * x**2)

        r = rr * cc**(1/3)
        dr = rr * x * np.sqrt(ss) * (cc-1)
        return r, dr
    return func


def chebyshev(c, n):
    """
    Radial symmetric Chebyshev shape radius as function of theta angle.

    Parameters
    ----------
    c : float or ndarray
        The deformation parameter.
    n : int
        The order of the Chebyshev polynomial.

    Returns
    -------
    func(x) : function
        A function that returns the radius and first derivative of the
        radius at the theta angles determined by Gaussian quadrature x.
    """
    c = c[..., None]
    def func(x):
        dn = n**2
        a = 1 + 1.5*c**2 * (4*dn - 2)/(4*dn - 1)
        if n%2 == 0:
            a -= 3*c*(1 + 0.25*c**2) / (dn-1) - 0.25*c**3 / (9*dn-1)

        dx = np.arccos(x)
        A = 1 + c * np.cos(n * dx)
        r = a**(-2/3) * A**2
        dr = -c * n * np.sin(n * dx) / A
        return r, dr
    return func


def gen_chebyshev(c, ng=60):
    """
    Radius as a function of theta angle for generalized chebyshev shape.

    Parameters
    ----------
    c : 1darray or ndarray
        An array containing the deformation parameters for each term in the
        Chebyshev polynomial, starting with order 0. The length of c is the
        order of the Chebyshev polynomial + 1.
    ng : int, optional
        Number of sample points and weights for Gauss-Legendre quadrature.
        It must be >= 1. Default is 60.

    Returns
    -------
    func(x) : function
        A function that returns the radius and first derivative of the
        radius at the theta angles determined by Gaussian quadrature x.
    """
    c = c[..., None]
    def func(x):
        n = np.arange(c.shape[-2])[:, None]

        x2, w = np.polynomial.legendre.leggauss(ng)
        dx2 = np.arccos(x2)
        A = 1 + np.sum(c * np.cos(n*dx2), axis=-2)
        ens = np.sum(c * n * np.sin(n*dx2), axis=-2)

        V = np.sum(w * A * np.sin(dx2)* A * (np.sin(dx2)*A + ens*x2), axis=-1)
        a = (3/4) * V[..., None]

        dx = np.arccos(x)
        A = 1 + np.sum(c * np.cos(n*dx), axis=-2)
        ens = np.sum(c * n * np.sin(n*dx), axis=-2)

        dr = -ens / A
        r = a**(-2/3) * A**2
        return r, dr
    return func


###############################################################################
# The following are specific raindrop shapes as a function of equivalent volume
# diameter.

def dropshape_CB90(D):
    """
    Raindrop shape according to Chuang and Beard (1990).

    Interpolates the coefficients of the generalized Chebyshev shape that are
    described in table 1 in Chuang and Beard (1990).

    Parameters
    ----------
    D : float or ndarray
        The equal-volume diameter in meter.

    Returns
    -------
    func(x) : function
        A function that returns the radius and first derivative of the
        radius at the theta angles determined by Gaussian quadrature x.
    """

    r = np.array([1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 8.,
                  9.])*1e-3
    C0 = np.array([28, 72, 134, 211, 297, 388, 481, 573, 665, 755, 843, 930,
                   1014, 1187, 1328])
    C1 = np.array([30, 70, 118, 180, 247, 309, 359, 401, 435, 465, 472, 487,
                   492, 482, 403])
    C2 = np.array([83, 210, 385, 592, 816, 1042, 1263, 1474, 1674, 1863, 2040,
                   2207, 2364, 2650, 2899])
    C3 = np.array([22, 57, 100, 147, 188, 221, 244, 255, 258, 251, 240, 222,
                   199, 148, 106])
    C4 = np.array([-3, -6, -5, 4, 24, 53, 91, 137, 187, 242, 299, 358, 419,
                   543, 662])
    C5 = np.array([2, 7, 17, 32, 52, 75, 99, 121, 141, 157, 168, 175, 178, 171,
                   153])
    C6 = np.array([1, 3, 6, 10, 13, 15, 15, 11, 4, -7, -21, -37, -56, -100,
                   -146])
    C7 = np.array([0, 0, 1, 3, 8, 15, 25, 36, 48, 61, 73, 84, 93, 107, 111])
    C8 = np.array([0, -1, -3, -5, -8, -12, -16, -19, -21, -21, -20, -16, -12,
                   2, 18])
    C9 = np.array([0, 0, -1, -1, -1, 0, 2, 6, 11, 17, 25, 34, 43, 64, 81])
    C10 = np.array([0, 1, 1, 2, 4, 7, 10, 13, 17, 21, 24, 27, 30, 32, 31])

    c = np.stack((-C0, C1, -C2, C3, C4, -C5, C6, C7, C8, -C9, C10))
    c = np.moveaxis(interp1d(r, c*1e-4, bounds_error=False)(D), 0, -1)
    return gen_chebyshev(c)


def dropshape_Th07(D):
    """
    Raindrop shape according to Thurai et al. (2007).

    Computes the raindrop axis ratio according to the empirical formula by
    Thurai et al. (2007) assuming a spheroid drop shape.

    Parameters
    ----------
    D : float or ndarray
        The equal-volume diameter in meter.

    Returns
    -------
    func(x) : function
        A function that returns the radius and first derivative of the
        radius at the theta angles determined by Gaussian quadrature x.
    """
    ar1 = 1.173 - 0.5165e3*D + 0.4698e6*D**2 - 0.1317e9*D**3 - 8.5e9*D**4
    ar2 = 1.065 - 62.5*D - 3.99e3*D**2 + 7.66e5*D**3 - 4.095e7*D**4
    c1 = D < 0.7e-3
    c2 = D <= 1.5e-3
    c3 = D <= 10e-3
    ar = c1*1. + ~c1*c2*ar1 + ~c2*ar2
    ar = np.where(c3, ar, np.nan)
    c = 1/ar
    return spheroid(c)


def dropshape_An99(D):
    """
    Raindrop shape according to Andsager et al. (1999).

    Computes the raindrop axis ratio according to the empirical formula by
    Andsager et al. (1999) assuming a spheroid drop shape.

    Parameters
    ----------
    D : float or ndarray
        The equal-volume diameter in meter.

    Returns
    -------
    func(x) : function
        A function that returns the radius and first derivative of the
        radius at the theta angles determined by Gaussian quadrature x.
    """
    ar1 = 1.0048 + 0.57*D - 2.628e4*D**2 + 3.682e6*D**3 - 1.677e8*D**4
    ar2 = 1.012 - 14.4*D - 1.03e4*D**2
    c1 = D < 1.1e-3
    c2 = D <= 4.4e-3
    c3 = D <= 7e-3
    ar = ~c1 * c2 * ar1 + (c1 + ~c2) * ar2
    ar = np.where(c3, ar, np.nan)
    c = 1/ar
    return spheroid(c)


###############################################################################
if __name__ == '__main__':
    """
    Example of usage.
    """
    D = np.asarray([2e-3, 4e-3])
    shape = dropshape_CB90(D)
    cheb = shape(np.polynomial.legendre.leggauss(100)[0])
    print(cheb)
