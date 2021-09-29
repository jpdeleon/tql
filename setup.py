#!/usr/bin/env python
from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="tql",
    packages=["tql"],
    version="0.1.1",
    author="Jerome de Leon",
    author_email="jpdeleon@astron.s.u-tokyo.ac.jp",
    url="https://github.com/jpdeleon/tql",
    #license=["GNU GPLv3"],
    description="TESS QuickLook plot generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # package_dir={"tql": "tql"},
    scripts=["scripts/tql", "scripts/rank_tls", "scripts/rank_gls"],
    # include_package_data=True,
    # package_data={'': ['*.csv', '*.cfg']},
    keywords=["TESS", "exoplanets", "stars"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python",
    ],
    #install_requires=['chronos @ http://github.com/jpdeleon/chronos/tarball/master#egg=chronos'],
    dependency_links=[
        "http://github.com/jpdeleon/chronos/tarball/master#egg=chronos"
    ],
)
