import rpcjs.elements as html
from rpcjs.binding import get_element_size, set_attribute, bind

from olympus.dashboard.queue_pages.inspect import InspectQueue
from olympus.dashboard.queue_pages.utilities import objective_array
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
        self.y2label = None
        self.colors = None
        self.data = None
        self.graph_id = 'graph'
        self.sample = None
        self.columns = None
        self.width = 640
        self.height = 480
        self.mark = None
        self.shape = None
        self.mark_options = [
            ('none', None),
            ('area', alt.Chart.mark_area),
            ('bar', alt.Chart.mark_bar),
            ('boxplot', alt.Chart.mark_boxplot),
            ('circle', alt.Chart.mark_circle),
            ('errorbar', alt.Chart.mark_errorbar),
            ('errorband', alt.Chart.mark_errorband),
            ('line', alt.Chart.mark_line),
            ('point', alt.Chart.mark_point),
            ('square', alt.Chart.mark_square),
        ]

    def set_attribute_callback(self, name):
        def set_attribute(data):
            debug(f'set {name} to {data}')
            if data == 0:
                data = None
            setattr(self, name, data)

            self.make_graph()
        return set_attribute

    def set_size(self, data):
        self.width = data['width'] * 0.80
        self.height = data['height'] * 0.70

    def guess_datatype(self, label, mark):
        from datetime import datetime

        if self.sample is None:
            return 'quantitative'

        data = self.sample[label]
        if isinstance(data, str):
            return 'nominal'
        if isinstance(data, float):
            return 'quantitative'
        if isinstance(data, int):
            if mark in {'errorband', 'boxplot', 'errorbar'}:
                return 'quantitative'

            return 'ordinal'

        if isinstance(data, datetime):
            return 'temporal'

        return 'quantitative'

    def make_graph(self):
        debug('new graph')
        get_element_size('graph_container', self.set_size)

        if self.xlabel is None or self.ylabel is None or self.mark is None:
            return

        mark = self.mark_options[self.mark][0]
        xlabel = self.columns[self.xlabel]
        ylabel = self.columns[self.ylabel]
        kwargs = {
            'x': alt.X(xlabel, type=self.guess_datatype(xlabel, mark)),
            'y': alt.Y(ylabel, type=self.guess_datatype(ylabel, mark))
        }

        if self.y2label is not None:
            y2label = self.columns[self.y2label]
            kwargs['y2'] = alt.Y2(y2label)

        if self.shape is not None:
            shape = self.columns[self.shape]
            kwargs['shape'] = alt.Shape(shape, type=self.guess_datatype(shape, mark))

        if self.colors is not None:
            colors = self.columns[self.colors]
            kwargs['color'] = alt.Color(colors, type=self.guess_datatype(colors, mark))

        mark_method = self.mark_options[self.mark][1]

        chart = mark_method(alt.Chart(self.data)).encode(
            **kwargs
        ).interactive().properties(
            width=self.width,
            height=self.height
        )

        set_attribute(self.graph_id, 'srcdoc', html.altair_plot(chart, with_iframe=False))

    def form(self, options, sample):
        import json
        options.insert(0, None)
        mark_opt = list(map(lambda i: i[0], self.mark_options))

        form_html = html.div(
            html.div(
                'Mark',
                html.select_dropdown(mark_opt, 'mark')),
            html.div(
                'X axis',
                html.select_dropdown(options, 'x-label')),
            html.div(
                'Y axis',
                html.select_dropdown(options, 'y-label')),
            html.div(
                'Y2 axis',
                html.select_dropdown(options, 'y2-label')),
            html.div(
                'Colors',
                html.select_dropdown(options, 'colors')),
            html.div(
                'Shape',
                html.select_dropdown(options, 'shape')),
            html.div(
                'Data Example',
                html.pre(json.dumps(sample, indent=2))
            )
        )

        # Request callbacks when those two values change
        bind('mark'    , 'change', self.set_attribute_callback('mark'), property='selectedIndex')
        bind('x-label' , 'change', self.set_attribute_callback('xlabel'), property='selectedIndex')
        bind('y-label' , 'change', self.set_attribute_callback('ylabel'), property='selectedIndex')
        bind('y2-label', 'change', self.set_attribute_callback('y2label'), property='selectedIndex')
        bind('colors'  , 'change', self.set_attribute_callback('colors'), property='selectedIndex')
        bind('shape'   , 'change', self.set_attribute_callback('shape'), property='selectedIndex')
        return form_html

    def show_queue(self, queue, namespace, delimiter=None):
        args = (queue, namespace)
        kwargs = {}
        if delimiter:
            kwargs['delimiter'] = delimiter

        messages = self.client.messages(*args, **kwargs)
        data = list(objective_array(messages))

        if len(data) == 0:
            return "No Metric"

        self.sample = data[-1]
        self.columns = list(data[-1].keys())
        self.data = alt.Data(values=data)
        form = self.form(self.columns, self.sample)
        return html.div_row(
            html.div_col(form, size=2, style="height: 100vh;"),
            html.div_col(html.iframe("", id=self.graph_id), id='graph_container', style='width: 100vh; height: 100vh;'))
