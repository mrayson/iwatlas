"""
Build with:
     python setup.py build_ext --inplace

See this site for building on windows-64:
	https://github.com/cython/cython/wiki/CythonExtensionsOnWindows

See this example for packaging cython modules:
        https://github.com/thearn/simple-cython-example/blob/master/setup.py
"""

from setuptools import setup #, find_packages
from setuptools.dist import Distribution

class BinaryDistribution(Distribution):
    def is_pure(self):
        return False


setup(
    name = "iwatlas",
    packages=[
        'iwatlas',
        ],
    version="0.0.1",
    description='Internal Wave climatological atlas driver routines',
    author='Matt Rayson',
    author_email='matt.rayson@uwa.edu.au',
    license='LICENSE',
    distclass=BinaryDistribution,
)
