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
            'olympus.dashboard',
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
        setup_requires=['setuptools'],
        tests_require=['pytest', 'flake8', 'codecov', 'pytest-cov'],
        entry_points={
            'console_scripts': [
                'olympus = olympus.baselines.launch:main',
                'olympus-dash = olympus.dashboard.main:main',
                'olympus-mongo = olympus.distributed.mongo:start_mongod',
                'olympus-port = olympus.distributed.network:get_free_port',
                'olympus-hpo-worker = olympus.hpo.worker:main'
            ]
        },
        extras_require={
            'geffnet': ['geffnet==0.9.3'],
            'rl': ['gym', 'procgen', 'atari_py'],
            'dash': ['plotly', 'plotly-express', 'altair', 'eventlet', 'flask-socketio'],
            'nlp': ['transformers'],
            'parallel': ['psycopg2-binary', 'pymongo']
            # > pip install git+git://github.com/Delaunay/track
            # 'track': ['']
        }
    )
