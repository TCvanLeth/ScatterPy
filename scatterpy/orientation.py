#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orientation averaging
=====================

This module contains functions for calculating the scattering characteristics
for collections of rotational symmetric scatterers with variable orientations
averaged over their orientations with statistical weights.

Functions
---------
.. autosummary::
    calc_SZ_oa
    calc_s_oa
    sph_uniform_pdf
    sph_gauss_pdf
"""

import numpy as np
from scipy.integrate import quad

from scatterpy.tmatrix import calc_T, calc_S, calc_Z


def calc_SZ_oa(T, theta0, theta, phi0, phi, pdf, n_alpha=5):
    """
    Compute the S and Z matrices using variable orientation scatterers.
    Averages over orientation.

    Parameters
    ----------
    T : ndarray
        The T matrix or array of matrices.
    theta0 : float or ndarray
        Incoming beam angle in rad.
    theta : float or ndarray
        Outgoing beam angle in rad.
    phi0 : float or ndarray
        Incoming beam angle in rad.
    phi : float or ndarray
        Outgoing beam angle in rad.
    pdf : function
        A function that returns the value of a spherical Jacobian-
        normalized PDF at x (radians). It is assumed to be normalized for the
        interval [0, pi].
    n_alpha : int
        The number of discrete alpha angles to perform the averaging over.

    Returns
    -------
    S : ndarray
        The amplitude matrix or array of matrices.
    Z : ndarray
        The phase matrix of array of matrices.
    """
    alpha = np.linspace(0, 2 * np.pi, n_alpha+1)[:-1]
    beta, weight = get_points_and_weights(pdf, left=0, right=np.pi)

    T = T[..., None, None, :, :, :, :, :]
    theta = theta[..., None, None]
    theta0 = theta0[..., None, None]
    phi = phi[..., None, None]
    phi0 = phi0[..., None, None]

    beta = beta[None, :]
    weight = weight[None, :, None, None]
    alpha = alpha[:, None]

    S = calc_S(T, theta0, theta, phi0, phi, alpha, beta)
    Z = calc_Z(S)

    aw = 1.0/(n_alpha * weight.sum())
    S = (S * weight).sum(axis=(-4, -3)) * aw
    Z = (Z * weight).sum(axis=(-4, -3)) * aw
    return S, Z


def calc_S_oa(T, theta0, theta, phi0, phi, pdf, n_alpha=5):
    """
    Compute the S matrices using variable orientation scatterers.
    Averages over orientation.

    Parameters
    ----------
    T : ndarray
        The T matrix or array of matrices.
    theta0 : float or ndarray
        Incoming beam angle in rad.
    theta : float or ndarray
        Outgoing beam angle in rad.
    phi0 : float or ndarray
        Incoming beam angle in rad.
    phi : float or ndarray
        Outgoing beam angle in rad.
    pdf : function
        A function that returns the value of a spherical Jacobian-
        normalized PDF at x (radians). It is assumed to be normalized for the
        interval [0, pi].
    n_alpha : int
        The number of discrete alpha angles to perform the averaging over.

    Returns
    -------
    S : ndarray
        The amplitude matrix or array of matrices.
    """
    alpha = np.linspace(0, 2 * np.pi, n_alpha+1)[:-1]
    beta, weight = get_points_and_weights(pdf, left=0, right=np.pi)

    T = T[..., None, None, :, :, :, :, :]
    theta = theta[..., None, None]
    theta0 = theta0[..., None, None]
    phi = phi[..., None, None]
    phi0 = phi0[..., None, None]

    beta = beta[None, :]
    weight = weight[None, :, None, None]
    alpha = alpha[:, None]

    S = calc_S(T, theta0, theta, phi0, phi, alpha, beta)

    aw = 1.0/(n_alpha * weight.sum())
    S = (S * weight).sum(axis=(-4, -3)) * aw
    return S


###############################################################################
def get_points_and_weights(w_func, left=-1.0, right=1.0, n_points=3, n=4096):
    """
    Quadrature points and weights for a weighting function.
    using Gautschi's method.

    (based on code by J. Leinonen)
    """
    dx = (right-left)/n
    z = np.linspace(left+0.5*dx, right-0.5*dx, n)
    w = w_func(z) * dx
    zw = z * w

    # Gautschi
    a = np.empty(n_points)
    b = np.empty(n_points-1)
    a[0] = zw.sum() / w.sum()
    p_prev = np.ones(z.shape)
    p = z - a[0]
    for j in range(1, n_points):
        p_norm = w @ p**2
        a[j] = zw @ p**2 / p_norm
        b[j-1] = p_norm / (w @ p_prev**2)
        p_new = z * p - a[j] * p - b[j-1] * p_prev
        p_prev = p
        p = p_new
    b = np.sqrt(b)

    J = np.diag(a)
    J += np.diag(b, k=-1)
    J += np.diag(b, k=1)

    points, v = np.linalg.eigh(J)
    ind = points.argsort()
    points = points[ind]
    weights = (v[0, :]**2 * w.sum())[ind]
    return points, weights


def sph_uniform_pdf():
    """
    Uniform PDF for orientation averaging.

    Returns
    -------
    pdf(x) : function
        A function that returns the value of the spherical Jacobian-normalized
        uniform PDF. It is normalized for the interval [0, pi].

    (based on code by J. Leinonen)
    """
    norm_const = 1

    def pdf(x):
        return norm_const * np.sin(x)

    # ensure that the integral over the distribution equals 1
    norm_dev = quad(pdf, 0, np.pi)[0]
    norm_const /= norm_dev
    return pdf


def sph_gauss_pdf(std=0.2):
    """
    Gaussian PDF for orientation averaging.

    Parameters
    ----------
    std : float, optional
        The standard deviation of the PDF (the mean is always taken to be 0).

    Returns
    -------
    pdf(x) : function
        A function that returns the value of the spherical Jacobian-
        normalized Gaussian PDF with the given STD at x (radians). It is
        normalized for the interval [0, pi].

    (based on code by J. Leinonen)
    """
    norm_const = 1

    def pdf(x):
        return norm_const*np.exp(-0.5 * (x/std)**2) * np.sin(x)

    # ensure that the integral over the distribution equals 1
    norm_dev = quad(pdf, 0, np.pi)[0]
    norm_const /= norm_dev
    return pdf


###############################################################################
if __name__ == '__main__':
    """
    Example of usage.
    """
    from scatter import shapes

    attrs = {'quantity':'diameter', 'unit':'m'}
    D = np.arange(4e-3, 5e-3, 1e-4) # equivalent sphere diameter

    wl = np.arange(8e-3, 9.1e-3, 1e-4)

    wl = 8E-3  # wavelength of incident light (same unit as radius)
    mr = np.asarray([1.5+0.02j])  # refractive index

    sfunc = shapes.dropshape
    wfunc = sph_gauss_pdf()

    theta0 = np.deg2rad(56) # incomming beam angle
    theta = np.deg2rad(65) # outgoing beam angle
    phi0 = np.deg2rad(114) # incoming beam angle
    phi = np.deg2rad(128) # outgoing beam angle

    T = calc_T(D, wl, mr, sfunc=sfunc)
    S, Z = calc_SZ_oa(T, theta0, theta, phi0, phi, wfunc)
    S *= wl / (2 * np.pi)
    print(S)
