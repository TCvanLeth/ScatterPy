# -*- coding: utf-8 -*-
import setuptools

with open("README.md", 'r') as fh:
    long_description = fh.read()

setuptools.setup(
        name="scatterpy",
        version="0.1.1",
        author="T.C. van Leth",
        author_email="tommy.vanleth@wur.nl",
        description="Electromagnetic scattering using T-matrix approach",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/TCvanLeth/ScatterPy",
        packages=setuptools.find_packages(),
        install_requires=[
                'numpy',
                'scipy',
        ],
        classifiers=[
                "Programming Language :: Python :: 3",
                "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
                "Operating System :: OS Independent",
        ],
)