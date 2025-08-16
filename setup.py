#!/usr/bin/env python

"""The setup script."""

from typing import List

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md", encoding="utf-8") as history_file:
    history = history_file.read()

# Core dependencies moved from requirements.txt
install_requires: List[str] = [
    "geopandas>=0.13.0",
    "pandas>=1.5.0",
    "shapely>=1.8.0",
    "pyrosm>=0.6.0",
    "osmnx>=1.6.0",
    "numpy>=1.21.0",
    "pyyaml>=6.0.0",
    "pyarrow>=10.0.0",
    "contextily>=1.3.0",
]

setup_requires = [
    "pytest-runner>=2.11.1",
]

tests_require = [
    "pytest>=3.4.2",
    "pytest-cov>=2.6.0",
]

development_requires = [
    # general
    "bumpversion>=0.5.3",
    "pip>=9.0.1",
    "watchdog>=0.8.3",
    # docs
    "myst-parser>=0.18.0",
    "Sphinx>=7.0.0",
    "sphinx_rtd_theme>=2.0.0",
    "autodocsumm>=0.1.10",
    # style check and formatting
    "ruff>=0.1.0",
    "pre-commit>=3.0.0",
    # distribute on PyPI
    "twine>=1.10.0",
    "wheel>=0.30.0",
    # Advanced testing
    "coverage>=4.5.1",
    "tox>=2.9.1",
]

setup(
    author="MIT Data To AI Lab",
    author_email="dailabmit@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    description="A data preprocessing pipeline for generating georeferenced datasets for synthetic low-voltage grid infrastructure modeling in the United States.",
    extras_require={
        "test": tests_require,
        "dev": development_requires + tests_require,
    },
    install_package_data=True,
    install_requires=install_requires,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="gridtracer electrical-grid geospatial preprocessing energy-modeling",
    name="gridtracer",
    packages=find_packages(include=["gridtracer", "gridtracer.*"]),
    python_requires=">=3.9",
    setup_requires=setup_requires,
    test_suite="tests",
    tests_require=tests_require,
    url="https://github.com/DAI-Lab/gridtracer",
    version="0.1.0.dev0",
    zip_safe=False,
)
