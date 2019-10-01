# -*- coding: utf-8 -*-
"""
Created on Fri May 20 15:26:41 2016

@author: tcvanleth
"""
import numpy as np

def surf_spheroid(e):
    """
    calculate shape factor for spheroid particles
    ar is the axis ratio between the horizontal and rotational axis
    """
    if e < 1:
        E = np.sqrt(1-e**2)
        R = np.sqrt(0.5*(e**(2/3) + e**(-1/3)*np.arcsin(E)/E))
        return 1/R
    else:
        E = np.sqrt(1-1/(e**2))
        R = np.sqrt(0.25*(2*e**(2/3) + e**(-4/3)*np.log((1+E)/(1-E))/E))
        return 1/R


def surf_cilinder(e):
    """
    calculate the shape factor for cilindrical particles
    ar is the ratio between the diameter and the length
    """
    return (1.5/e)**(1/3) / np.sqrt((e+2)/(2*e))


def surf_chebyshev(n, c):
    """
    calculate the shape factor for Chebyshev particles

    n is the degree of the chebyshev polynomial
    c is the deformation parameter
    """
    ng = 60
    x, w = np.polynomial.legendre.leggauss(ng)

    dx = np.arccos(x)
    A = 1 + c * np.cos(n*dx)
    ENS = c * n * np.sin(n*dx)

    S = np.sum(w * A * np.sqrt(A**2 + ENS**2))
    V = np.sum(w * A * np.sin(dx)*A * (np.sin(dx)*A + ENS * x))

    RV = (V*3/4)**(1/3)
    rat = RV / np.sqrt(S*0.5)

    dn = n**2
    a = 1 + 1.5*c**2 * (4*dn - 2)/(4*dn - 1)
    if n%2 == 0:
        a -= 3*c*(1 + 0.25*c**2) / (dn-1) - 0.25*c**3 / (9*dn-1)
    r0v = a**(1/3)
    return rat, r0v


def surf_drop(c):
    """
    calculate the shape factor for generalized Chebyshev particles

    c is an array containing the coefficients of the polynomial expansion
    of the particle shape
    """
    nc = 11
    c = np.zeros(nc)
    c[0] = -0.0481
    c[1] = 0.0359
    c[2] = -0.1263
    c[3] = 0.0244
    c[4] = 0.0091
    c[5] = -0.0099
    c[6] = 0.0015
    c[7] = 0.0025
    c[8] = -0.0016
    c[9] = -0.0002
    c[10] = 0.0010
    n= np.arange(len(c))

    ng = 60
    x, w = np.polynomial.legendre.leggauss(ng)

    dx = np.arccos(x)
    A = 1 + np.sum(c * np.cos(n*dx))
    ENS = np.sum(c * n * np.sin(n*dx))

    S = np.sum(w * A * np.sqrt(A**2 + ENS**2))
    V = np.sum(w * A * np.sin(dx)*A * (np.sin(dx)*A + ENS * x))

    RV = (V*3/4)**(1/3)
    rat = RV / np.sqrt(S*0.5)
    r0v = 1/RV
    return rat, r0v


def legendre(ng, shape, ar):
    # based on n_gauss
    if shape == 'cilinder':
        ng1 = ng//2
        ng2 = ng - ng1
        XX = -np.cos(np.arctan(ar))
        x1, w1 = np.polynomial.legendre.leggauss(ng1)
        x2, w2 = np.polynomial.legendre.leggauss(ng2)

        x = np.zeros(2*ng)
        w = np.zeros(2*ng)

        w[:ng1] = 0.5 * (XX+1) * w1
        x[:ng1] = 0.5 * (XX+1) * x1 + 0.5*(XX-1)

        w[ng1:ng] = -0.5 * XX * w2
        x[ng1:ng] = -0.5 * XX * x2 + 0.5*XX

        w[:ng-1:-1] = w[:ng]
        x[:ng-1:-1] = -x[:ng]
    else:
        x, w = np.polynomial.legendre.leggauss(2*ng)
    return x, w


def rsp_cilinder(x, radi_ev, e):
    """
    radi_ev is the equal volume radius
    e is the ratio of the diameter to the heigth
    """
    height = radi_ev * ((2/(3*e**2))**(1/3))
    radius = height * e

    si = np.sqrt(1 - x**2)
    cond = si/x > radius/height
    rad = np.where(cond, radius/si, -height/x)
    r_theta = np.where(cond, radius*x/si**2, height*si/x**2)

    r = rad*2
    dr = -r_theta / rad
    return r, dr
