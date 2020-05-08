import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import datetime
import json
import time
import os
import copy
import re
import pprint

import numpy

from msgqueue.backends import new_client
from sspace.space import compute_identity

import yaml

import xarray

from olympus.observers.msgtracker import METRIC_QUEUE
from olympus.observers.observer import Metric
from olympus.hpo.parallel import (
    make_remote_call, RESULT_QUEUE, WORK_QUEUE, WORK_ITEM, HPO_ITEM, RESULT_ITEM)
from olympus.hpo import HPOptimizer, Fidelity
from olympus.utils.functional import flatten
from olympus.studies.variance.main import fetch_results, register, wait


IDENTITY_SIZE = 16


class Interrupt(Exception):
    pass


@dataclass
class InterruptingMetric(Metric):
    frequency_epoch: int = 1
    # Interrupt after Checkpointing
    priority: int = -100
    epoch: int = field(default=0)

    def on_end_epoch(self, task, epoch, context):
        print(f'Interrupting Epoch {epoch} {self.frequency_epoch}')
        raise Interrupt()

    def value(self):
        return {}


def generate(num_experiments, num_repro, objective, variables, defaults, resumable):

    # TODO: Add a resume test as well
    # Run 5 times full
    # Run 5 times half stopped then resumed

    # NOTE TO TEST: make the uid dependent of repetition number, otherwise there will be collisions in
    # checkpoints
    # NOTE Set the checkpointer buffer to 0 to make sure checkpoints are done
    # NOTE Not all tasks need checkpoints. Do not launch checkpoint tests for them.

    configs = dict()
    for variable in variables:
        configs[variable] = []
        for seed in range(1, num_experiments + 1):
            for repetition in range(1, num_repro + 1):
                kwargs = copy.copy(defaults)
                kwargs[variable] = int(seed)
                kwargs['repetition'] = repetition
                uid = compute_identity(kwargs, IDENTITY_SIZE)
                kwargs.pop('repetition')
                kwargs['uid'] = uid
                configs[variable].append(kwargs)
                if resumable:
                    kwargs = copy.copy(kwargs)
                    kwargs['_interrupt'] = True
                    kwargs['repetition'] = repetition
                    kwargs.pop('uid')
                    uid = compute_identity(kwargs, IDENTITY_SIZE)
                    kwargs.pop('repetition')
                    kwargs['uid'] = uid
                    configs[variable].append(kwargs)

    return configs 


def test(data, num_experiments, num_repro, objective, variables, resumable):

    failures = []

    for variable in variables:
        var_data = data.sel(seed=variable)
        for seed_i in range(num_experiments):
            k = (seed_i * num_repro) * (2 if resumable else 1)
            reference = var_data.isel(order=k)[objective]
            for repro_i in range(num_repro):
                k = (seed_i * num_repro + repro_i) * (2 if resumable else 1)
                a = reference.values
                b = var_data.isel(order=k)[objective].values
                if ((a == b) | (numpy.isnan(a) & numpy.isnan(b))).all() and False:
                    print(variable, seed_i + 1, repro_i + 1, 'ok')
                else:
                    failures.append((variable, seed_i + 1, repro_i + 1, a, b))
                    print(variable, seed_i + 1, repro_i + 1, 'fail')

    if not failures:
        print('Success!')
        return

    print('Failures:')
    for failure in failures:
        variable, seed, repro, a, b = failure
        print(variable, seed, repro)
        print(a)
        print(b)


def save_results(namespace, data, save_dir):
    with open(f'{save_dir}/repro_{namespace}.json', 'w') as f:
        f.write(json.dumps(data.to_dict()))


def load_results(namespace, save_dir):
    with open(f'{save_dir}/repro_{namespace}.json', 'r') as f:
        data = xarray.Dataset.from_dict(json.loads(f.read()))

    return data


def run(uri, database, namespace, function, num_experiments, num_repro, objective,
        variables, defaults, params, resumable, sleep_time=60, save_dir='.'):

    client = new_client(uri, database)

    defaults.update(dict(list(variables.items()) + list(params.items())))

    configs = generate(num_experiments, num_repro, objective, list(sorted(variables)), defaults, resumable)
    register(client, function, namespace, configs)

    wait(client, namespace, sleep=sleep_time)

    data = fetch_results(client, namespace, configs, list(sorted(variables)), params, defaults)

    save_results(namespace, data, save_dir)

    test(data, num_experiments, num_repro, objective, variables, resumable)


def run_from_config_file(uri, database, namespace, config_file, **kwargs):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    module = __import__('.'.join(config['function'].split('.')[:-1]), fromlist=[''])
    config['function'] = getattr(module, config['function'].split('.')[-1])
    config.update(kwargs)
    return run(uri, database, namespace, **config)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--uri', default='mongo://127.0.0.1:27017', type=str)
    parser.add_argument('--database', default='olympus', type=str)
    parser.add_argument('--namespace', default=None, type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--sleep-time', default=60, type=int)
    parser.add_argument('--num-experiments', default=10, type=int)
    parser.add_argument('--num-repro', default=10, type=int)
    parser.add_argument('--save-dir', default='.', type=str)
    args = parser.parse_args(args)

    namespace = args.namespace
    if namespace is None:
        namespace = '.'.join(os.path.basename(args.config).split('.')[:-1])

    run_from_config_file(
        args.uri, args.database, namespace, args.config,
        num_experiments=args.num_experiments,
        num_repro=args.num_repro,
        sleep_time=args.sleep_time, 
        save_dir=args.save_dir)


if __name__ == '__main__':
    main()
