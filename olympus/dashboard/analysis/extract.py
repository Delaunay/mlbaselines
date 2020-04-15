from track.structure import Trial
from track.serialization import to_json

import pandas as pd


def flatten_trials_metrics(trials, protocol):
    """Flatten trials metrics"""
    metrics = []

    for trial_id in trials:
        t = Trial()
        t.uid = trial_id
        trial = to_json(protocol.get_trial(t)[0])

        values = {'uid': trial['uid']}

        for metric_name, metric_values in trial['metrics'].items():

            new_metrics = {}
            for k, v in metric_values.items():
                new_metrics[k] = v

            values[metric_name] = new_metrics

        metrics.append(pd.DataFrame(values))

    extracted_data = pd.concat(metrics, sort=True)
    return extracted_data
