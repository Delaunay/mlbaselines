from olympus.dashboard.queue_pages.inspect import InspectQueue
from olympus.dashboard.queue_pages.utilities import objective_array
from olympus.dashboard.elements import altair_plot
from olympus.dashboard.dash import bind, set_attribute
import olympus.dashboard.elements as html

from olympus.utils import debug

import altair as alt


class MetricQueue(InspectQueue):
    base_path = 'queue/metric'

    def __init__(self, client):
        super(MetricQueue, self).__init__(client)
        self.title = 'Metric Queue'
        self.xlabel = None
        self.ylabel = None
        self.data = None
        self.graph_id = 'graph'
        self.columns = None

    def set_xlabel(self, data):
        debug(f'set x-label to {data}')
        self.xlabel = self.columns[data]
        self.make_graph()

    def set_ylabel(self, data):
        debug(f'set y-label to {data}')
        self.ylabel = self.columns[data]
        self.make_graph()

    def make_graph(self):
        debug('new graph')

        if self.xlabel is None or self.ylabel is None:
            return

        chart = alt.Chart(self.data).mark_line().encode(
            x=f'{self.xlabel}:Q',
            y=f'{self.ylabel}:Q'
        )

        set_attribute(self.graph_id, 'srcdoc', altair_plot(chart))

    def form(self, options):
        form_html = html.div(
            html.div(
                html.header('X axis', level=4),
                html.select_dropdown(options, 'x-label')),
            html.div(
                html.header('Y axis', level=4),
                html.select_dropdown(options, 'y-label')))

        # Request callbacks when those two values change
        bind('x-label', 'change', self.set_xlabel, property='selectedIndex')
        bind('y-label', 'change', self.set_ylabel, property='selectedIndex')

        return form_html

    def show_queue(self, queue, namespace):
        messages = self.client.messages(queue, namespace)
        data = list(objective_array(messages))
        assert len(data) > 0
        self.columns = list(data[0].keys())
        self.data = alt.Data(values=data)
        form = self.form(self.columns)
        return html.div_row(
            html.div_col(form, size=2),
            html.div_col(f'<iframe width="100%" height="600px" id="{self.graph_id}"></iframe>', size=8))
