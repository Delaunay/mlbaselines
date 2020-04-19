from olympus.dashboard.queue_pages.inspect import InspectQueue
from olympus.dashboard.queue_pages.utilities import extract_last_results
from olympus.dashboard.plots.hyperparameter_importance import importance_heatmap_altair, marginals_altair
import olympus.dashboard.elements as html
from olympus.dashboard.elements import altair_plot
from olympus.dashboard.analysis.hpfanova import FANOVA

import pandas as pd


class FANVOAQueue(InspectQueue):
    base_path = 'queue/fanova'

    def __init__(self, client):
        super(FANVOAQueue, self).__init__(client)
        self.title = 'FANVOA'

    def show_queue(self, queue, namespace):
        messages = self.client.messages(queue, namespace)

        data, columns = extract_last_results(messages)

        # Make a list of all hyper parameters
        columns = set(columns)
        columns.discard('uid')
        columns.discard('epoch')
        columns.discard('objective')

        columns = list(columns)
        columns.sort()
        rcolumns = list(reversed(columns))
        # --

        # Select a specific fidelity
        all_data = pd.DataFrame(data)
        min_epoch = all_data['epoch'].min()
        all_data = all_data[all_data['epoch'] == min_epoch]
        # --

        fanova = FANOVA(
            all_data,
            hp_names=columns,
            objective='objective',
            hp_space={
                a: 'uniform(0, 1)' for a in columns
            })

        imp = importance_heatmap_altair(fanova)
        marginals = marginals_altair(fanova)

        page = html.div(
            html.div(
                html.header('Importance', level=4),
                altair_plot(imp),
                style="height:300px;"),
            html.div(
                html.header('Marginals', level=4),
                altair_plot(marginals),
                style="height:300px;"))

        return page
