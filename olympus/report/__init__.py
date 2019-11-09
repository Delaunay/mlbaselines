import json

from orion.storage.base import Storage
from orion.core.utils import flatten

from olympus.utils import get_storage as resolve_storage, show_dict

import pandas as pd
import plotly.offline as py
import plotly.express as px
import plotly.io as pio

import math


class Report:
    def __init__(self, storage_uri, experiment_name):
        self.storage_config = resolve_storage(storage_uri)
        self.storage_type = self.storage_config.pop('type')
        self.storage = Storage(of_type=self.storage_type, **self.storage_config)

        self.experiment_name = experiment_name

        print(self.storage.fetch_experiments({}))
        experiments = self.storage.fetch_experiments(dict(name=experiment_name))
        assert experiments, 'Did not find any experiments'
        self.experiment = experiments[0]

        self.trials = list(self.storage.fetch_trials(uid=self.experiment['_id']))
        self.trials.sort(key=lambda x: x.objective.value, reverse=True)

        self.values = self.extract_results()

    def extract_results(self):
        values = []

        for trial in self.trials:
            data = {
                'id': trial.id
            }

            for result in trial.results:
                data[result.name] = result.value

            values.append(data)
        return pd.DataFrame(values)

    def show_all(self):
        print('=' * 80)
        print('    Experiment: ', self.experiment['name'])
        show_dict(flatten.flatten(self.experiment['space']))

        for i, t in enumerate(self.trials):
            print('   ', '=' * 80)
            obj = t.objective
            print(f'    > Trial {i} ({obj.name}: {obj.value})')
            show_dict(flatten.flatten(t.params), indent=4)

            for r in t.results:
                print(f'{r.name:>30}: {r.value}')
        print('=' * 80)

    def show_distribution(self, metric):
        print(self.values)
        fig = px.histogram(self.values, x=metric,  marginal="rug", nbins=10)
        pio.write_image(fig, '/Tmp/img.png')


if __name__ == '__main__':
    report = Report('legacy:pickleddb:full_test.pkl', 'classification_mnist_logreg_sgd_none_glorot_uniform')
    report.show_distribution('validation_accuracy')


