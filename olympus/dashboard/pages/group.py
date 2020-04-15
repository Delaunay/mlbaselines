import dash_html_components as html

import pandas as pd

from olympus.dashboard.render import TrialGroupRender
from olympus.dashboard.pages.page import Page
from olympus.dashboard.render.simple_plot import SimplePlot
from olympus.dashboard.render.altair_plot import AltairPlot
from olympus.dashboard.analysis.extract import flatten_trials_metrics

_group_cache = {}


class GroupPage(Page):
    @staticmethod
    def route():
        return '/group/(?P<group_id>[a-zA-Z0-9_]*)'

    def __init__(self, protocol):
        self.protocol = protocol
        self.group = None
        # The plot need to the instantiated right now because
        # its events need to be forwarded ASAP for them to work
        # self.plot = AltairPlot(dataframe=pd.DataFrame())
        self.plot = SimplePlot(dataframe=pd.DataFrame())

    def events(self, app):
        self.plot.events(app)

    def render(self, app, group_id):
        global _group_cache

        if group_id in _group_cache:
            self.group, self.plot.dataframe = _group_cache[group_id]
        else:
            groups = self.protocol.fetch_groups({'uid': group_id})

            if not groups:
                return f'(group_id: {group_id}) does not exist'

            self.group = groups[0]
            self.plot.dataframe = flatten_trials_metrics(self.group.trials, self.protocol)
            _group_cache[group_id] = (self.group, self.plot.dataframe)

        self.plot.choices = list(self.plot.dataframe.columns)

        return html.Div(
            className='row',
            children=[
                html.Div(
                    className='group-view col-md-auto',
                    children=[TrialGroupRender(self.group).render(app)]),
                html.Div(
                    className='group-plot col',
                    children=self.plot.render(app),
                )])


