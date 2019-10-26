#!/usr/bin/env python

from setuptools import setup


if __name__ == '__main__':
    setup(
        name='olympus',
        version='0.0.0',
        description='Compendium of ML Models',
        author='Pierre Delaunay, Xavier Bouthillier',
        packages=[
            'olympus',
            'olympus.classification',
            'olympus.distributed',
            'olympus.utils',
        ],
        install_requires=[
            'dataclasses',
            'typing',
            'pytorch'
        ],
        setup_requires=['setuptools'],
        tests_require=['pytest', 'flake8', 'codecov', 'pytest-cov'],
    )
