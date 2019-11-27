import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

import plotly.graph_objs as go

from olympus.report.base import DOMComponent
from olympus.report.render.utils import prettify_name


class SimplePlot(DOMComponent):
    options = [{'label': i, 'value': i} for i in ['linear', 'log']]

    def __init__(self, dataframe, dom_id='simple-plot'):
        self.id = dom_id
        self.dataframe = dataframe
        self.choices = list(self.dataframe.columns)

    def axis_options(self, name, placeholder):
        return html.Div(
            className='col-sm-4',
            children=[
                html.Div(
                    dcc.Dropdown(
                        className='dropdown',
                        placeholder=placeholder,
                        id=f'{name}-column',
                        options=[{'label': prettify_name(i), 'value': i} for i in self.choices],
                        style={'vertical-align': 'middle'}
                    ),
                    style={'display': 'inline-block', 'width': '50%', 'vertical-align': 'middle'}),
                dcc.RadioItems(
                    id=f'{name}-type',
                    options=self.options,
                    value='linear',
                    style={'display': 'inline-block', 'padding': '5px'}
                )])

    def plot_options(self):
        return html.Div(
            className='form-group row',
            children=[
                self.axis_options('xaxis', 'X Axis'),
                self.axis_options('yaxis', 'Y Axis')
            ])

    def events(self, app):
        inputs = [
            Input('xaxis-column', 'value'), Input('yaxis-column', 'value'),
            Input('xaxis-type', 'value'),   Input('yaxis-type', 'value')
        ]

        app.callback(
            Output(self.id, 'figure'), inputs
        )(self.on_render_event)

    def preprocess_data(self, xaxis_col, yaxis_col, group_by='uid'):
        lines = []

        if group_by:
            groups = self.dataframe.groupby([group_by])

            for k, _ in groups.indices.items():
                group = groups.get_group(k)

                lines.append(go.Scatter(
                    x=group[xaxis_col],
                    y=group[yaxis_col],
                    mode='lines',
                    name=k[:10]
                ))
        else:
            group = self.dataframe

            lines.append(go.Scatter(
                x=group[xaxis_col],
                y=group[yaxis_col],
                mode='markers',
                marker={
                    'size': 15,
                    'opacity': 0.5,
                    'line': {'width': 0.5, 'color': 'white'}
                }
            ))

        return lines

    def render(self, app):
        return html.Div(
            id=f'{self.id}-simple-plot',
            className='col',
            style={'width': '100%'},
            children=[
                self.plot_options(),
                dcc.Graph(id=self.id)
            ])

    def on_render_event(self, xaxis_col, yaxis_col, xaxis_type, yaxis_type):
        if xaxis_col is None or yaxis_col is None:
            return {}

        xaxis_type = xaxis_type.lower()
        yaxis_type = yaxis_type.lower()

        lines = self.preprocess_data(xaxis_col, yaxis_col)

        return {
            'data': lines,
            'layout': go.Layout(
                xaxis={
                    'title': prettify_name(xaxis_col),
                    'type': 'linear' if xaxis_type == 'linear' else 'log'
                },
                yaxis={
                    'title': prettify_name(yaxis_col),
                    'type': 'linear' if yaxis_type == 'linear' else 'log'
                },
                margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
                height=450,
                hovermode='closest'
            )
        }
