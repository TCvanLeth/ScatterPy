#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ScatterPy T-matrix simulation of electromagnetic scattering by nonspherical
particles.

Copyright (C) 2019 Thomas C. van Leth
-------------------

This module contains functions to numerically calculate scattering amplitude
and phase matrices for arbitrary rotationally symmetric particles using the
T-matrix method.

These routines are based on the original FORTRAN routines by M.I. Mishchenko (1998).

Functions
---------
.. autosummary::
    calc_T
    calc_S
    calc_Z
"""

import gc

import numpy as np

from scatterpy import matstack
from scatterpy import shapes
from scatterpy import special


def calc_T(D, wl, mr, sfunc=None, n_maxorder=None, n_gauss=None,
           nm_max=100, ng_max=500, **kwargs):
    """
    Calculate the T-matrix given arrays of particle characteristics.

    Parameters
    ----------
    D : ndarray or float
        Equivalent-volume diameter of the scattering particle.
    wl : ndarray or float
        Wavelength of the scattered electromagnetic radiation. Should be given
        in the same units as D.
    mr : ndarray or complex
        Complex index of refraction for the scattering particle.
    sfunc : function, optional
        Function of D that returns a function of the horizontal radius of the
        scattering particle as a function of vertical angle. If no function is
        provided default to a perfect sphere.
    n_maxorder : int, optional
        Tuning parameter determining the largest order to use for Bessel,
        Hankel and Wigner d functions. Warning: setting this too large will
        greatly affect the amount of memory required. By default, this value is
        determined by autotuning.
    n_gauss : int, optional
        Number of sample points and weights for the Gauss-Legendre quadrature
        used to evaluate the shape of the particle. By default, this value is
        determined by autotuning.
    nm_max : int, optional
        Maximum value for n_maxorder when autotuning. Default is 100.
    ng_max : int, optional
        Maximum value of n_gauss when autotuning. Default is 500.

    Returns
    -------
    T : ndarray
        The T-matrix or array of matrices. Last dimensions are
        n_maxorder, n_gauss, n_gauss, 2, 2.
    """
    if sfunc is None:
        sfunc = lambda x: shapes.spheroid(np.array(1.0))

    # Calculate relative particle size.
    x_ev = np.pi * D / wl
    shape = sfunc(D)

    if n_maxorder is None or n_gauss is None:

        # Select the most 'extreme' parameter combination from the arrays,
        # which is likely to take the longest to converge.
        x_ev_max = np.nanmax(x_ev)
        shape_max = sfunc(np.nanmax(D))
        args = (x_ev_max, np.nanmax(mr), shape_max)

        # tune n_max parameter
        xmin = int(max(4, np.max(x_ev_max + 4.05 * x_ev_max**(1/3))))
        pfunc = lambda x: {'n_maxorder': x, 'n_gauss': x*2}
        pars = tune(calc_T_inner, args, xmin, nm_max, pfunc, 'n_maxorder', **kwargs)

        # tune n_gauss parameter
        pfunc = lambda x: {'n_maxorder': pars['n_maxorder'], 'n_gauss': x}
        pars = tune(calc_T_inner, args, pars['n_gauss'] + 1, ng_max, pfunc, 'n_gauss', **kwargs)
        n_maxorder, n_gauss = pars['n_maxorder'], pars['n_gauss']

    return calc_T_inner(x_ev, mr, shape, n_maxorder=n_maxorder, n_gauss=n_gauss)


def calc_T_inner(x_ev, mr, shape, n_maxorder=7, n_gauss=15, check=False):
    """
    Compute the T-matrix for a given particle shape function.

    Parameters
    ----------
    x_ev : ndarray or float
        Dimensionless equivalent volume diameter of the particle relative to
        the wavelength of the scattered radiation.
    mr : ndarray or complex
        Complex index of refraction for the scattering particle.
    shape : function
        Function that returns the horizontal radius of the scattering
        particle as a function of vertical angle.
    n_maxorder : int, optional
        Tuning parameter determining the largest order to use for Bessel,
        Hankel and Wigner d functions. Warning: setting this too large will
        greatly affect the amount of memory required. Defaults to 7.
    n_gauss : int, optional
        Number of sample points and weights for the Gauss-Legendre quadrature
        used to evaluate the shape of the particle. Defaults to 15.
    check : bool, optional
        When set to true, only the first slice of the T-matrix is evaluated
        and convergence checks are returned. This is usually only used when
        called by the autotuning function. Defaults to false.

    Returns
    -------
    T : ndarray
        The T-matrix or array of matrices. Last dimensions are
        n_maxorder, n_gauss, n_gauss, 2, 2.
    Qext : float, optional
        Convergence check
    Qsca : float, optional
        Convergence check
    """
    # n_gauss dependent
    x, w = np.polynomial.legendre.leggauss(2 * n_gauss)
    r, dr = shape(x)
    rr = (w * r)[..., None, :, None, None]
    del w

    z = np.sqrt(r) * x_ev[..., None]
    z_mr = z * mr[..., None]
    ddr = (dr / z)[..., None, :, None, None]
    drc = (dr / z_mr)[..., None, :, None, None]
    del r, dr

    # n_max dependent
    if check:
        m_maxorder = 0
    else:
        m_maxorder = n_maxorder

    n = np.arange(1, n_maxorder + 1)
    an = n * (n + 1)
    an2 = np.sqrt((2 * n + 1) / an)
    rr = rr * an2[:, None] * an2[None, :]
    ddr = ddr * an[:, None]
    drc = drc * an[None, :]
    del an, an2, n

    s = (np.arange(m_maxorder + 1)[:, None]**2 / (1 - x**2))[..., None, None]
    rs = rr * np.sqrt(s)

    # Compute the spherical Bessel functions.
    J, dJ = _bessel(z, special.sph_jn, n_maxorder)
    Jc, dJc = _bessel(z_mr, special.sph_jn, n_maxorder)

    # Compute the Hankel functions.
    H, dH = map(lambda x, y: x + 1j * y, (J, dJ), _bessel(z, special.sph_yn, n_maxorder))
    del z, z_mr

    # create wigner d-matrix
    dv1, dv2 = special.wigner_d(x, n_maxorder, m_maxorder)
    wig_d = np.swapaxes(matstack.combo(dv1, dv2, dv1, dv2), 1, 2)
    del dv1, dv2, x
    gc.collect()

    # create vector spherical functions
    E = [wig_d[3] + wig_d[0] * s,
         wig_d[1] + wig_d[2]]
    F = [wig_d[1] * ddr,
         wig_d[2] * drc,
         wig_d[0] * ddr,
         wig_d[0] * drc]
    del ddr, drc, s, wig_d
    gc.collect()

    # create Q and RgQ matrices
    B = matstack.combo(H, dH, Jc, dJc)[..., None, :, :, :]
    Q = _calc_Q(B, rr, rs, mr, E, F)
    B = matstack.combo(J, dJ, Jc, dJc)[..., None, :, :, :]
    RgQ = _calc_Q(B, rr, rs, mr, E, F)
    del B, rr, rs, E, F
    gc.collect()

    # solve T*Q = -RgQ for T
    T = matstack.solve(Q, -RgQ).swapaxes(-2, -1)  # <-- transpose

    # convergence checks
    if check:
        D_n1 = np.arange(3, 2 * n_maxorder + 3, 2)
        Tdiag = T[0].diagonal()

        Qext = np.sum(D_n1 * (Tdiag.real[:n_maxorder] + Tdiag.real[n_maxorder:]))
        Qsca = np.sum(D_n1 * (np.abs(Tdiag[:n_maxorder])**2 + np.abs(Tdiag[n_maxorder:])**2))
        return Qext, Qsca

    return matstack.unstack(T)


def calc_S(T, theta0, theta, phi0, phi, alpha, beta):
    """
    Compute the S-matrix given the T-matrix and the beam geometry.

    Parameters
    ----------
    T : ndarray
        The T-matrix.
    theta0 : float or ndarray.
        Incoming beam zenith angle in rad.
    theta : float or ndarray
        Outgoing beam zenith angle in rad.
    phi0 : float or ndarray
        Incoming beam azimuth angle in rad.
    phi : float or ndarray
        Outgoing beam azimuth angle in rad.
    alpha : float or ndarray
        Drop canting euler angle in rad (rotation around z axis).
    beta : float or ndarray
        Drop canting euler angle in rad (rotation around x' axis).

    Returns
    -------
    S : ndarray
        The scattering amplitude matrix or array of matrices.
    """
    # compute matrices R and R1^(-1), EQ. (13)
    B = _calc_B(alpha, beta)

    AL0 = _calc_A(phi0, theta0)
    phi02, theta02 = _angle_trans(phi0, theta0, alpha, beta)
    AP0 = _calc_A(phi02, theta02)
    R0 = np.swapaxes(AP0, -1, -2) @ (B @ AL0)

    AL1 = _calc_A(phi, theta)
    phi2, theta2 = _angle_trans(phi, theta, alpha, beta)
    AP1 = _calc_A(phi2, theta2)
    R1 = np.swapaxes(AP1, -1, -2) @ (B @ AL1)

    # compute wigner d-matrices
    n_max = T.shape[-3]
    dv1, dv2 = special.wigner_d(np.cos(theta2), n_max)
    dv01, dv02 = special.wigner_d(np.cos(theta02), n_max)
    m = np.arange(n_max + 1)[:, None]

    dv1 = dv1 * m / np.sqrt(1 - np.cos(theta2)**2)[..., None, None]
    dv01 = dv01 * m / np.sqrt(1 - np.cos(theta02)**2)[..., None, None]
    D = matstack.combo(dv1, dv2, dv01, dv02)

    # other stuff
    nn = np.arange(1, n_max + 1)[None, :]
    n = np.arange(1, n_max + 1)[:, None]
    CAL = 1j**(nn - n - 1) * np.sqrt((2 * n + 1) * (2 * nn + 1) /
                                     (n * nn * (n + 1) * (nn + 1)))

    m = np.arange(n_max+1)[:, None, None]
    diff_phi2 = (phi2 - phi02)[..., None, None, None]
    CN1 = CAL[None, :, :] * 2*np.cos(m * diff_phi2)
    CN2 = CAL[None, :, :] * 2*np.sin(m * diff_phi2)
    CN1[..., 0, :, :] = CAL

    # compute matrix S
    S00 = np.sum((T[..., 0, 0] * D[0] + T[..., 1, 0] * D[2] +
                  T[..., 0, 1] * D[1] + T[..., 1, 1] * D[3]) * CN1,
                 axis=(-1, -2, -3))
    S01 = np.sum((T[..., 0, 0] * D[1] + T[..., 1, 0] * D[3] +
                  T[..., 0, 1] * D[0] + T[..., 1, 1] * D[2]) * CN2,
                 axis=(-1, -2, -3))
    S10 = -np.sum((T[..., 0, 0] * D[2] + T[..., 1, 0] * D[0] +
                   T[..., 0, 1] * D[3] + T[..., 1, 1] * D[1]) * CN2,
                  axis=(-1, -2, -3))
    S11 = np.sum((T[..., 0, 0] * D[3] + T[..., 1, 0] * D[1] +
                  T[..., 0, 1] * D[2] + T[..., 1, 1] * D[0]) * CN1,
                 axis=(-1, -2, -3))
    S = np.stack([np.stack([S00, S10], axis=-1),
                  np.stack([S01, S11], axis=-1)], axis=-1)
    S = np.linalg.solve(R1, S @ R0)
    return S


def calc_Z(S):
    """
    Compute the Z-matrix given the S-matrix.

    Parameters
    ----------
    S : ndarray
        The scattering amplitude matrix.

    Returns
    -------
    Z : ndarray
        The scattering phase matrix
    """
    Z = np.zeros(S.shape[:-2] + (4, 4), complex)

    Z[..., 0, 0] = 0.5 * (S[..., 0, 0] * S[..., 0, 0].conj() +
                          S[..., 0, 1] * S[..., 0, 1].conj() +
                          S[..., 1, 0] * S[..., 1, 0].conj() +
                          S[..., 1, 1] * S[..., 1, 1].conj())
    Z[..., 0, 1] = 0.5 * (S[..., 0, 0] * S[..., 0, 0].conj() -
                          S[..., 0, 1] * S[..., 0, 1].conj() +
                          S[..., 1, 0] * S[..., 1, 0].conj() -
                          S[..., 1, 1] * S[..., 1, 1].conj())
    Z[..., 0, 2] = -1. * (S[..., 0, 0] * S[..., 0, 1].conj() +
                          S[..., 1, 1] * S[..., 1, 0].conj())
    Z[..., 0, 3] = +1j * (S[..., 0, 0] * S[..., 0, 1].conj() -
                          S[..., 1, 1] * S[..., 1, 0].conj())
    Z[..., 1, 0] = 0.5 * (S[..., 0, 0] * S[..., 0, 0].conj() +
                          S[..., 0, 1] * S[..., 0, 1].conj() -
                          S[..., 1, 0] * S[..., 1, 0].conj() -
                          S[..., 1, 1] * S[..., 1, 1].conj())
    Z[..., 1, 1] = 0.5 * (S[..., 0, 0] * S[..., 0, 0].conj() -
                          S[..., 0, 1] * S[..., 0, 1].conj() -
                          S[..., 1, 0] * S[..., 1, 0].conj() +
                          S[..., 1, 1] * S[..., 1, 1].conj())
    Z[..., 1, 2] = -1. * (S[..., 0, 0] * S[..., 0, 1].conj() -
                          S[..., 1, 1] * S[..., 1, 0].conj())
    Z[..., 1, 3] = +1j * (S[..., 0, 0] * S[..., 0, 1].conj() +
                          S[..., 1, 1] * S[..., 1, 0].conj())
    Z[..., 2, 0] = -1. * (S[..., 0, 0] * S[..., 1, 0].conj() +
                          S[..., 1, 1] * S[..., 0, 1].conj())
    Z[..., 2, 1] = -1. * (S[..., 0, 0] * S[..., 1, 0].conj() -
                          S[..., 1, 1] * S[..., 0, 1].conj())
    Z[..., 2, 2] = +1. * (S[..., 0, 0] * S[..., 1, 1].conj() +
                          S[..., 0, 1] * S[..., 1, 0].conj())
    Z[..., 2, 3] = -1j * (S[..., 0, 0] * S[..., 1, 1].conj() +
                          S[..., 1, 0] * S[..., 0, 1].conj())
    Z[..., 3, 0] = +1j * (S[..., 1, 0] * S[..., 0, 0].conj() +
                          S[..., 1, 1] * S[..., 0, 1].conj())
    Z[..., 3, 1] = +1j * (S[..., 1, 0] * S[..., 0, 0].conj() -
                          S[..., 1, 1] * S[..., 0, 1].conj())
    Z[..., 3, 2] = -1j * (S[..., 1, 1] * S[..., 0, 0].conj() -
                          S[..., 0, 1] * S[..., 1, 0].conj())
    Z[..., 3, 3] = +1. * (S[..., 1, 1] * S[..., 0, 0].conj() -
                          S[..., 0, 1] * S[..., 1, 0].conj())
    return Z


def tune(func, args, xmin, xmax, pfunc, name, **kwargs):
    """
    Tune computational function by gradually increasing precision and checking
    for convergence.
    """
    Qext = 0
    Qsca = 0
    for ix in range(xmin, xmax+1):
        pars = pfunc(ix)
        Qext1 = Qext
        Qsca1 = Qsca
        Qext, Qsca = func(*args, check=True, **pars)
        if np.isclose(Qext1, Qext, **kwargs) and \
           np.isclose(Qsca1, Qsca, **kwargs):
            break
    else:
        raise Exception('convergence is not obtained for '+name+'='+str(xmax))
    return pars


###############################################################################
# T-matrix calculation subroutines
def _bessel(z, func, n_max):
    """
    Compute the spherical bessel functions and first derivatives.
    """
    n = np.arange(n_max+1)
    J = np.zeros(z.shape + (n_max+1,), z.dtype)
    for i in n:
        J[..., i] = func(i, z)
    dJ = J[..., :-1] - n[1:] * J[..., 1:] / z[..., None]
    return J[..., 1:], dJ


def _calc_Q(B, rr, rs, mr, E, F):
    """
    Calculate Q/RgQ matrix describing the relation between incident/scattered
    field and internal field.
    """
    mr = mr[..., None, None, None]
    A_01 = np.sum(rr * (E[0] * B[2] + F[0] * B[0]), axis=-3)
    A_10 = np.sum(rr * (E[0] * B[1] + F[1] * B[0]), axis=-3)
    A_00 = np.sum(rs * (E[1] * B[0]), axis=-3)
    A_11 = np.sum(rs * (E[1] * B[3] + F[2] * B[1] + F[3] * B[2]), axis=-3)

    Q = np.zeros(A_10.shape + (2, 2), complex)
    Q[..., 0, 0] = (mr * A_10 - A_01) * -1j
    Q[..., 1, 1] = (mr * A_01 - A_10) * 1j
    Q[..., 0, 1] = (mr * A_00 + A_11)
    Q[..., 1, 0] = (mr * A_11 + A_00)
    return matstack.stack(Q).swapaxes(-2, -1)


###############################################################################
# S-matrix computation subroutines
def _calc_A(phi, theta):
    """
    Compute the matrices AL and AL1, Eq.(14).
    """
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    shp = tuple(max(a, b) for a, b in zip(phi.shape, theta.shape))
    A = np.zeros(shp+(3, 2), float)
    A[..., 0, 0] = cos_theta * cos_phi
    A[..., 0, 1] = -sin_phi
    A[..., 1, 0] = cos_theta * sin_phi
    A[..., 1, 1] = cos_phi
    A[..., 2, 0] = -sin_theta
    A[..., 2, 1] = 0
    return A


def _calc_B(alpha, beta):
    """
    Compute the matrix BETA, Eq.(21).
    """
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)

    # compute matrix BETA, EQ. (21)
    shp = tuple(max(a, b) for a, b in zip(alpha.shape, beta.shape))
    B = np.zeros(shp + (3, 3), float)
    B[..., 0, 0] = cos_alpha * cos_beta
    B[..., 0, 1] = sin_alpha * cos_beta
    B[..., 0, 2] = -sin_beta
    B[..., 1, 0] = -sin_alpha
    B[..., 1, 1] = cos_alpha
    B[..., 1, 2] = 0
    B[..., 2, 0] = cos_alpha * sin_beta
    B[..., 2, 1] = sin_alpha * sin_beta
    B[..., 2, 2] = cos_beta
    return B


def _angle_trans(phi, theta, alpha, beta):
    """
    Incident and scattered angle transformation.
    """
    cos_phi = np.cos(phi - alpha)
    sin_phi = np.sin(phi - alpha)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)

    theta2 = np.arccos(cos_theta * cos_beta + sin_theta * sin_beta * cos_phi)
    phi2 = np.arctan2((sin_theta * sin_phi),
                      (cos_beta * sin_theta * cos_phi - sin_beta * cos_theta))
    return phi2, theta2


###############################################################################
if __name__ == '__main__':
    """
    Example of usage.
    """

    radius = np.asarray([5., 6.])  # equivalent sphere radius (meter)
    wl = 3.1415925  # wavelength of incident light (same unit as radius)
    mr = np.asarray([1.5+0.02j, 1.8+0.02j])  # refractive index

    alpha = np.asarray([np.deg2rad(145)]) # Drop canting euler angle in rad (rotation around z axis).
    beta = np.asarray([np.deg2rad(52)]) # # Drop canting euler angle in rad (rotation around x' axis).
    theta0 = np.asarray([np.deg2rad(56), np.deg2rad(60)]) # incomming beam zenith angle
    theta = np.asarray([np.deg2rad(65), np.deg2rad(70)]) # outgoing beam zenith angle
    phi0 = np.asarray([np.deg2rad(114), np.deg2rad(120)]) # incoming beam azimuth angle
    phi = np.asarray([np.deg2rad(128), np.deg2rad(130)]) # outgoing beam azimuth angle

    sfunc = lambda x: shapes.gen_chebyshev(np.array([-0.0481, 0.0359, -0.1263, 0.0244,
                                           0.0091, -0.0099, 0.0015, 0.0025,
                                           -0.0016, -0.0002, 0.0010]))
    T = calc_T(radius*2, wl, mr, rtol=0.001)
    S = calc_S(T, theta0, theta, phi0, phi, alpha, beta)
    Z = calc_Z(S)
    print(S)
