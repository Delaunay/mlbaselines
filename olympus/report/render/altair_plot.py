import io
import traceback

import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

from olympus.report.base import DOMComponent
from olympus.report.render.utils import prettify_name

import altair as alt
from vega_datasets import data


class AltairPlot(DOMComponent):
    def __init__(self, dataframe, dom_id='altair-plot'):
        self.id = dom_id
        self.dataframe = dataframe
        self.choices = list(self.dataframe.columns)
        self.chart = None

    @property
    def columns(self):
        cols = [html.H3('Columns')]
        for p in self.choices:
            cols.append(html.Li(p))

        return html.Ul(cols, className='altair-columns col-md-auto', style={
            'display': 'inline-block',
            'padding': '10px',
            'margin': '10px'
        })

    @property
    def altair_input(self):
        return dcc.Textarea(id=f'{self.id}-input', className='col')

    def axis_options(self, name, placeholder):
        return html.Div(className='col', children=[dcc.Dropdown(
            className='dropdown',
            placeholder=placeholder,
            id=f'{name}-column',
            options=[{'label': prettify_name(i), 'value': i} for i in self.choices],
            style={'vertical-align': 'middle'}
        )])

    def plot_options(self):
        return html.Div(children=[
            html.Div(
                className='row altair-code',
                style={'padding': '10px', 'margin': '10px'},
                children=[
                    self.columns,
                    self.altair_input]),
            html.Div(
                className='row',
                children=[
                    dcc.Textarea(
                        id=f'{self.id}-output',
                        readOnly=True,
                        className='col',
                        style={'padding': '20px', 'margin': '20px'}),
                ]
            )
        ])

    def events(self, app):
        # chart.mark_point().encode(x='epoch',y='validation_loss')

        # Execute Altair Code
        inputs = [Input(f'{self.id}-input', 'value')]
        output = Output(f'{self.id}-output', 'value')
        app.callback(output, inputs)(self.execute)

        inputs = [Input(f'{self.id}-output', 'value')]
        output = Output(f'altair-plot-{self.id}', 'srcDoc')
        app.callback(output, inputs)(self.render_chart)

    def execute(self, altair_code):
        try:
            self.chart = None
            globs = {
                'alt': alt
            }
            locls = {
                'chart': alt.Chart(self.dataframe),
            }

            self.chart = eval(altair_code, globs, locls)
            return ''
        except:
            return traceback.format_exc()

    def preprocess_data(self, xaxis_col, yaxis_col, group_by='uid'):
        return

    def render(self, app):
        return html.Div(
            id=self.id,
            className='altair-plot-area',
            style={'width': '100%'},
            children=[
                # Plot Options
                self.plot_options(),
                # Plot Element
                html.Iframe(
                    id=f'altair-plot-{self.id}',
                    height='100%',
                    width='100%',
                    sandbox='allow-scripts',
                    style={
                        'border-width': '0px',
                        'position': 'absolute'
                    }
                )
            ])

    def render_chart(self, dummy):
        if self.chart:
            # Convert Chart to HTML
            chart_html = io.StringIO()
            self.chart.save(chart_html, 'html')
            return chart_html.getvalue()

        return ''
