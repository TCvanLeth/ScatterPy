Python module to calculate scattering amplitude and phase matrices of
rotationally symmetric particles based on the T-matrix method.

Based on the original Fortran T-matrix code by M.I. Mishchenko. (https://www.giss.nasa.gov/staff/mmishchenko/t_matrix.html). Includes orientation averaging schemes adapted from J. Leinonen (https://github.com/jleinonen/pytmatrix).

This is a pure Python reimplementation of the original routines using NumPy, not a Python wrapper around the Fortran code.

Currently spheroids, Chebyshev shapes and generalized chebyshev shapes are
implemented. Also includes several empirical raindrop shapes.

Download and install with pip:
`pip install scatterpy`

Examples of use
---------------

Single generalized Chebyshev particle:
```Python
import numpy as np
from scatterpy import *

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
```

Orientation averaged raindrops:
```Python
import numpy as np
from scatterpy import *

attrs = {'quantity':'diameter', 'unit':'m'}
D = np.arange(4e-3, 5e-3, 1e-4) # equivalent sphere diameter

wl = np.arange(8e-3, 9.1e-3, 1e-4)

wl = 8E-3  # wavelength of incident light (same unit as radius)
mr = np.asarray([1.5+0.02j])  # refractive index

sfunc = shapes.dropshape_CB90
wfunc = sph_gauss_pdf()

theta0 = np.deg2rad(56) # incomming beam zenith angle
theta = np.deg2rad(65) # outgoing beam zenith angle
phi0 = np.deg2rad(114) # incoming beam azimuth angle
phi = np.deg2rad(128) # outgoing beam azimuth angle

T = calc_T(D, wl, mr, sfunc=sfunc)
S, Z = calc_SZ_oa(T, theta0, theta, phi0, phi, wfunc)
S *= wl / (2 * np.pi)
print(S)
```

To evaluate the empirical raindrop shapes:
```Python
D = np.asarray([2e-3, 4e-3])
shape = dropshape_CB90(D)
cheb = shape(np.polynomial.legendre.leggauss(100)[0])
print(cheb)
```
