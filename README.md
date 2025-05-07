<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“DAI-Lab” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

<!-- Uncomment these lines after releasing the package to PyPI for version and downloads badges -->
<!--[![PyPI Shield](https://img.shields.io/pypi/v/syngrid.svg)](https://pypi.python.org/pypi/syngrid)-->
<!--[![Downloads](https://pepy.tech/badge/syngrid)](https://pepy.tech/project/syngrid)-->
[![Travis CI Shield](https://travis-ci.org/DAI-Lab/syngrid.svg?branch=master)](https://travis-ci.org/DAI-Lab/syngrid)
[![Coverage Status](https://codecov.io/gh/DAI-Lab/syngrid/branch/master/graph/badge.svg)](https://codecov.io/gh/DAI-Lab/syngrid)



# SynGrid

TBD

- Documentation: https://DAI-Lab.github.io/syngrid
- Homepage: https://github.com/DAI-Lab/syngrid

# Overview

TODO: Provide a short overview of the project here.

# Install

## Requirements

**SynGrid** has been developed and tested on [Python 3.5, 3.6, 3.7 and 3.8](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of a [virtualenv](https://virtualenv.pypa.io/en/latest/)
is highly recommended in order to avoid interfering with other software installed in the system
in which **SynGrid** is run.

These are the minimum commands needed to create a virtualenv using python3.6 for **SynGrid**:

```bash
pip install virtualenv
virtualenv -p $(which python3.6) syngrid-venv
```

Afterwards, you have to execute this command to activate the virtualenv:

```bash
source syngrid-venv/bin/activate
```

Remember to execute it every time you start a new console to work on **SynGrid**!

<!-- Uncomment this section after releasing the package to PyPI for installation instructions
## Install from PyPI

After creating the virtualenv and activating it, we recommend using
[pip](https://pip.pypa.io/en/stable/) in order to install **SynGrid**:

```bash
pip install syngrid
```

This will pull and install the latest stable release from [PyPI](https://pypi.org/).
-->

## Install from source

With your virtualenv activated, you can clone the repository and install it from
source by running `make install` on the `stable` branch:

```bash
git clone git@github.com:DAI-Lab/syngrid.git
cd syngrid
git checkout stable
make install
```

## Install for Development

If you want to contribute to the project, a few more steps are required to make the project ready
for development.

Please head to the [Contributing Guide](https://DAI-Lab.github.io/syngrid/contributing.html#get-started)
for more details about this process.

# Quickstart

In this short tutorial we will guide you through a series of steps that will help you
getting started with **SynGrid**.

TODO: Create a step by step guide here.

# What's next?

For more details about **SynGrid** and all its possibilities
and features, please check the [documentation site](
https://DAI-Lab.github.io/syngrid/).
