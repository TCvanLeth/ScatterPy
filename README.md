Python module to calculate scattering amplitude and phase matrices of
rotationally symmetric particles based on the T-matrix method.

Based on the original Fortran T-matrix code by M.I. Mishchenko. (https://www.giss.nasa.gov/staff/mmishchenko/t_matrix.html). Includes orientation averaging schemes adapted from J. Leinonen (https://github.com/jleinonen/pytmatrix).

This is a pure Python reimplementation of the original routines using NumPy, not a Python wrapper around the Fortran code.

Currently spheroids, Chebyshev shapes and generalized chebyshev shapes are
implemented. Also includes several empirical raindrop shapes.

Download and install with pip:
`pip install scatterpy`

To use:
`import scatterpy`
