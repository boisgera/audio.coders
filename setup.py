#!/usr/bin/env python
# coding: utf-8

# Python 3 Standard Library
import configparser

import os
import os.path
import shutil
import sys

# Pip Package Manager
try:
    import pip
    import setuptools
    import pkg_resources
except ImportError:
    error = "pip is not installed, refer to <{url}> for instructions."
    raise ImportError(error.format(url="http://pip.readthedocs.org"))

# Numpy
import numpy


def local(path):
    return os.path.join(os.path.dirname(__file__), path)

# Extra Third-Party Libraries
sys.path.insert(1, local(".lib"))
try:
    setup_requires = ["about>=4.0.0"]
    require = lambda *r: pkg_resources.WorkingSet().require(*r)
    require(*setup_requires)
    import about
except pkg_resources.DistributionNotFound:
    error = """{req!r} not found; install it locally with:

    pip install --target=.lib --ignore-installed {req!r}
"""
    raise ImportError(error.format(req=" ".join(setup_requires)))
import about

# This Package (Metadata only)
sys.path.insert(1, local("audio"))
import about_coders

#
# CYTHON and REST options management (from setup.cfg)
# ------------------------------------------------------------------------------
#
CYTHON = None
REST = None

setuptools.Distribution.global_options.extend([
    ("cython", None, "compile Cython files"),
    ("rest"  , None, "generate reST documentation")
])

def trueish(value):
    if not isinstance(value, str):
        return bool(value)
    else:
        value = value.lower()
        if value in ("y", "yes", "t", "true", "on", "1"):
            return True
        elif value in ("", "n", "no", "f", "false", "off", "0"):
            return False
        else:
            raise TypeError("invalid bool value {0!r}, use 'true' or 'false'.")

def import_CYTHON_REST_from_setup_cfg():
    global CYTHON, REST
    if os.path.isfile("setup.cfg"):
        parser = configparser.ConfigParser()
        parser.read("setup.cfg")
        try:
            CYTHON = trueish(parser.get("global", "cython"))
        except (configparser.NoOptionError, configparser.NoSectionError):
            pass
        try:
            REST = trueish(parser.get("global", "rest"))
        except (configparser.NoOptionError, configparser.NoSectionError):
            pass

import_CYTHON_REST_from_setup_cfg()

#
# Custom Developer Commands
# ------------------------------------------------------------------------------
#
def make_extension(name):
    include = dict(include_path=[numpy.get_include()])
    pyx_file = name.replace(".", "/") + ".pyx"
    c_file   = name.replace(".", "/") + ".c"
    if CYTHON:
        pkg_resources.require("Cython")
        import Cython
        from Cython.Build import cythonize
        return cythonize(pyx_file, **include)
    else:
        if os.path.exists(c_file):
            return [setuptools.Extension(name, 
                                         sources=[c_file],
                                         **include)]
        else:
            error = "file not found: {0!r}".format(c_file)
            raise IOError(error)

#
# Setup
# ------------------------------------------------------------------------------
#
info = dict(
  metadata     = about.get_metadata(about_coders),
  code         = dict(packages=setuptools.find_packages(),
                      ext_modules=make_extension("audio.coders")),
  data         = {},
  requirements = dict(install_requires="bitstream"),
  scripts      = {},
  plugins      = {},
  tests        = dict(test_suite="test.suite"),
)

if __name__ == "__main__":
    kwargs = {k:v for dct in info.values() for (k,v) in dct.items()}
    setuptools.setup(**kwargs)

