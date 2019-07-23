#!/usr/bin/env python
import os
import sys
import re

try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()

# Handle encoding
major, minor1, minor2, release, serial = sys.version_info
if major >= 3:
    def rd(filename):
        f = open(filename, encoding="utf-8")
        r = f.read()
        f.close()
        return r
else:
    def rd(filename):
        f = open(filename)
        r = f.read()
        f.close()
        return r

setup(
    name='tql',
    packages =['tql'],
    version="0.1.1",
    author='Jerome de Leon',
    author_email = 'jpdeleon@astron.s.u-tokyo.ac.jp',
    url = 'https://github.com/jpdeleon/tql',
    license = ['GNU GPLv3'],
    description ='TESS Quick Look plot generator',
    long_description=rd("README.md") + "\n\n"
                    + "---------\n\n",
    package_dir={"tql": "tql"},
    scripts=['scripts/make_tql'],
    #include_package_data=True,
    keywords=['TESS','exoplanets','stars'],
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python'
        ],
    install_requires = ['lightkurve','astropy','astroquery','astroplan','tqdm',
        'numpy','matplotlib','pandas','scipy','fbpca'],
)
