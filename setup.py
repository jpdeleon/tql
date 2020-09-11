#!/usr/bin/env python
from setuptools import setup, find_packages


def rd(filename):
    f = open(filename)
    r = f.read()
    f.close()
    return r


setup(
    name="tql",
    packages=["tql"],
    version="0.1.1",
    author="Jerome de Leon",
    author_email="jpdeleon@astron.s.u-tokyo.ac.jp",
    url="https://github.com/jpdeleon/tql",
    license=["GNU GPLv3"],
    description="TESS QuickLook plot generator",
    long_description=rd("README.md") + "\n\n" + "---------\n\n",
    # package_dir={"tql": "tql"},
    scripts=["scripts/tql", "scripts/rank_tls", "scripts/rank_gls"],
    # include_package_data=True,
    keywords=["TESS", "exoplanets", "stars"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python",
    ],
    dependency_links=[
        "http://github.com/jpdeleon/chronos/tarball/master#egg=chronos"
    ],
)
