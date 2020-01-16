#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ScatterPy T-matrix simulation of electromagnetic scattering by nonspherical
particles.

Copyright (C) 2019 Thomas C. van Leth
"""

import numpy as np


def stack(x):
    """
    Rearrange a 'ragged' stack of block matrices into a ragged stack of
    regular square matrices.
    """
    shp = x.shape
    y = np.zeros(shp[:-4] + (2*shp[-4], 2*shp[-3]), complex)
    for m in range(shp[-5]):
        m1 = 0 if m == 0 else (m-1)
        m2 = m1+shp[-3]
        y[..., m, 2*m1:m2, 2*m1:m2] = x[..., m, m1:, m1:, 0, 0]
        y[..., m, 2*m1:m2, m2:]     = x[..., m, m1:, m1:, 0, 1]
        y[..., m, m2:, 2*m1:m2]     = x[..., m, m1:, m1:, 1, 0]
        y[..., m, m2:, m2:]         = x[..., m, m1:, m1:, 1, 1]
    return y


def unstack(x):
    """
    Rearrange a 'ragged' stack of regular square matrices into a ragged stack
    of block matrices.
    """
    shp = x.shape
    y = np.zeros(shp[:-2] + (shp[-2] // 2, shp[-1] // 2) + (2, 2), complex)
    for m in range(shp[-3]):
        m1 = 0 if m == 0 else (m-1)
        m2 = m1+shp[-1]//2
        y[..., m, m1:, m1:, 0, 0] = x[..., m, 2*m1:m2, 2*m1:m2]
        y[..., m, m1:, m1:, 0, 1] = x[..., m, 2*m1:m2, m2:]
        y[..., m, m1:, m1:, 1, 0] = x[..., m, m2:, 2*m1:m2]
        y[..., m, m1:, m1:, 1, 1] = x[..., m, m2:, m2:]
    return y


def solve(a, b):
    """
    Solve a 'ragged' stack of matrices.
    """
    shp = a.shape
    x = np.zeros(shp, dtype=complex)
    for m in range(shp[-3]):
        m1 = 0 if m == 0 else (m-1)*2
        x[..., m, m1:, m1:] = np.linalg.solve(a[..., m, m1:, m1:],
                                              b[..., m, m1:, m1:])
    return x


def combo(A1, A2, B1, B2):
    """
    """
    C11 = A1[..., None] * B1[..., None, :]
    C12 = A1[..., None] * B2[..., None, :]
    C21 = A2[..., None] * B1[..., None, :]
    C22 = A2[..., None] * B2[..., None, :]
    return np.stack((C11, C12, C21, C22))
