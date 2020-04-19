from olympus.dashboard.queue_pages.inspect import InspectQueue
from olympus.dashboard.queue_pages.utilities import objective_array
from olympus.dashboard.elements import altair_plot
from olympus.dashboard.dash import bind, set_attribute, get_element_size
import olympus.dashboard.elements as html

from olympus.utils import debug

import altair as alt
alt.themes.enable('dark')


class MetricQueue(InspectQueue):
    base_path = 'queue/metric'

    def __init__(self, client):
        super(MetricQueue, self).__init__(client)
        self.title = 'Metric Queue'
        self.xlabel = None
        self.ylabel = None
        self.colors = None
        self.data = None
        self.graph_id = 'graph'
        self.sample = None
        self.columns = None
        self.width = 640
        self.height = 480

    def set_attribute_callback(self, name):
        def set_attribute(data):
            debug(f'set {name} to {data}')
            v = self.columns[data]

            if data == 0:
                v = None

            setattr(self, name, v)
            self.make_graph()
        return set_attribute

    def set_size(self, data):
        self.width = data['width'] * 0.80
        self.height = data['height'] * 0.80

    def guess_datatype(self, label):
        from datetime import datetime

        if self.sample is None:
            return 'quantitative'

        data = self.sample[label]
        if isinstance(data, str):
            return 'nominal'
        if isinstance(data, float):
            return 'quantitative'
        if isinstance(data, int):
            return 'ordinal'
        if isinstance(data, datetime):
            return 'temporal'
        return 'quantitative'

    def make_graph(self):
        debug('new graph')
        get_element_size('graph_container', self.set_size)

        if self.xlabel is None or self.ylabel is None:
            return

        kwargs = {
            'x': alt.X(self.xlabel, type=self.guess_datatype(self.xlabel)),
            'y': alt.Y(self.ylabel, type=self.guess_datatype(self.xlabel))
        }

        if self.colors is not None:
            kwargs['color'] = alt.Color(self.colors, type=self.guess_datatype(self.colors))

        chart = alt.Chart(self.data).mark_line().encode(
            **kwargs
        ).properties(
            width=self.width,
            height=self.height
        )

        set_attribute(self.graph_id, 'srcdoc', altair_plot(chart, with_iframe=False))

    def form(self, options, sample):
        import json
        options.insert(0, None)
        form_html = html.div(
            html.div(
                html.header('X axis', level=5),
                html.select_dropdown(options, 'x-label')),
            html.div(
                html.header('Y axis', level=5),
                html.select_dropdown(options, 'y-label')),
            html.div(
                html.header('Colors', level=5),
                html.select_dropdown(options, 'colors')),
            html.div(
                html.header('Data Example', level=5),
                html.pre(json.dumps(sample, indent=2))
            )
        )

        # Request callbacks when those two values change
        bind('x-label', 'change', self.set_attribute_callback('xlabel'), property='selectedIndex')
        bind('y-label', 'change', self.set_attribute_callback('ylabel'), property='selectedIndex')
        bind('colors', 'change', self.set_attribute_callback('colors'), property='selectedIndex')
        return form_html

    def show_queue(self, queue, namespace):
        messages = self.client.messages(queue, namespace)
        data = list(objective_array(messages))
        assert len(data) > 0

        self.sample = data[-1]
        self.columns = list(data[-1].keys())
        self.data = alt.Data(values=data)
        form = self.form(self.columns, self.sample)
        return html.div_row(
            html.div_col(form, size=2, style="height: 100vh;"),
            html.div_col(html.iframe("", id=self.graph_id), id='graph_container', style='width: 100vh; height: 100vh;'))
