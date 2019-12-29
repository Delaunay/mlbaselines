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
            'olympus.datasets.split',
            'olympus.distributed',
            'olympus.hpo',
            'olympus.metrics',
            'olympus.models',
            'olympus.models.inits',
            'olympus.optimizers',
            'olympus.optimizers.schedules',
            'olympus.reinforcement',
            'olympus.baselines',
            'olympus.tasks',
            'olympus.utils',
        ],
        install_requires=[
            'torch',
            'torchvision',
            # 'orion',
            'h5py',
            'psycopg2-binary',
            'gym',
            'tqdm',
            'pandas',
            'filelock'
        ],
        setup_requires=['setuptools'],
        tests_require=['pytest', 'flake8', 'codecov', 'pytest-cov'],
        entry_points={
            'console_scripts': [
                'olympus = olympus.baselines.launch:main',
                'olympus-dash = olympus.report.dashboard:main',
                'olympus-mongo = olympus.distributed.mongo:start_mongod',
                'olympus-port = olympus.distributed.network:get_free_port'
            ]
        },
        extras_require={
            'geffnet': ['geffnet==0.9.3'],
            'rl': ['gym', 'procgen', 'atari_py'],
            'dash': ['plotly', 'plotly-express', 'dash', 'altair'],
            # > pip install git+git://github.com/Delaunay/track
            # 'track': ['']
            # NVIDIA Apex would go there if there was a pip to give
            # note that this does not work you need to install it manually
            # > pip install git+git://github.com/NVIDIA/apex.git@606c3dcccd6ca70
            # 'float16': [
            #    'git+git://github.com/NVIDIA/apex.git@606c3dcccd6ca70'
            # ]
        }
    )
