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
            'torch',
            'torchvision',
            'orion'
        ],
        setup_requires=['setuptools'],
        tests_require=['pytest', 'flake8', 'codecov', 'pytest-cov'],
        entry_points={
            'console_scripts': [
                'olympus = olympus.scripts.launch:main',
            ]
        },
        extras_require={
            'geffnet': ['geffnet'],
            # NVIDIA Apex would go there if there was a pip to give
            'float16': []
        }
    )
