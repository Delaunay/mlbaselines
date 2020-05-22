#!/usr/bin/env python

from setuptools import setup


if __name__ == '__main__':
    extra_requires = {
        'geffnet': ['geffnet'],
        'rl': ['gym', 'procgen', 'atari_py'],
        'dash': ['plotly', 'plotly-express', 'altair', 'flask', 'eventlet', 'flask-socketio'],
        'nlp': ['transformers'],
        'parallel': ['psycopg2-binary', 'pymongo'],
        'testing': ['pytest', 'pytest-cov', 'codecov', 'coverage'],
        'analysis': ['xarray', 'scikit-learn', 'matplotlib', 'scikit-optimize']
    }

    all_packages = []
    for values in extra_requires.values():
        all_packages.extend(values)
    extra_requires['all'] = all_packages

    setup(
        name='olympus',
        version='0.0.0',
        description='Compendium of ML Models',
        author='Pierre Delaunay, Xavier Bouthillier',
        extras_require=extra_requires,
        packages=[
            'olympus',
            'olympus.accumulators',
            'olympus.datasets',
            'olympus.datasets.sampling',
            'olympus.datasets.split',
            'olympus.dashboard',
            'olympus.dashboard.analysis',
            'olympus.dashboard.plots',
            'olympus.dashboard.queue_pages',
            'olympus.distributed',
            'olympus.hpo',
            'olympus.metrics',
            'olympus.models',
            'olympus.models.inits',
            'olympus.observers',
            'olympus.optimizers',
            'olympus.optimizers.schedules',
            'olympus.reinforcement',
            'olympus.baselines',
            'olympus.tasks',
            'olympus.transforms',
            'olympus.utils',
            'olympus.utils.fp16',
            'olympus.utils.gpu',
            'olympus.utils.images',
            'olympus.studies',
            'olympus.studies.hpo',
            'olympus.studies.searchspace',
            'olympus.studies.variance'
        ],
        setup_requires=['setuptools'],
        install_require=['torch', 'filelock', 'torchvision'],
        tests_require=['pytest', 'flake8', 'codecov', 'pytest-cov'],
        entry_points={
            'console_scripts': [
                'olympus = olympus.baselines.launch:main',
                'olympus-dash = olympus.dashboard.main:main',
                'olympus-port = olympus.distributed.network:get_free_port',
                'olympus-hpo-worker = olympus.hpo.worker:main',
                'olympus-state-compare = olympus.utils.compare:main',
            ]
        }
    )
