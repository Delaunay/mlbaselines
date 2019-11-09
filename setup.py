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
            'olympus.accumulators',
            'olympus.datasets',
            'olympus.distributed',
            'olympus.hpo',
            'olympus.metrics',
            'olympus.models',
            'olympus.optimizers',
            'olympus.reinforcement',
            'olympus.scripts',
            'olympus.tasks',
            'olympus.transforms',
            'olympus.utils',
        ],
        install_requires=[
            'torch',
            'torchvision',
            'orion',
            'h5py',
            'psycopg2-binary',
            'gym',
            'tqdm'
        ],
        setup_requires=['setuptools'],
        tests_require=['pytest', 'flake8', 'codecov', 'pytest-cov'],
        entry_points={
            'console_scripts': [
                'olympus = olympus.scripts.launch:main',
            ]
        },
        extras_require={
            'geffnet': ['geffnet==0.9.3'],
            'rl': ['gym'],
            'dash': ['plotly-express', 'dash']
            # NVIDIA Apex would go there if there was a pip to give
            # note that this does not work you need to install it manually
            # > pip install git+git://github.com/NVIDIA/apex.git@606c3dcccd6ca70
            # 'float16': [
            #    'git+git://github.com/NVIDIA/apex.git@606c3dcccd6ca70'
            # ]
        }
    )
