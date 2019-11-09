
import json
import os

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

import numpy as np
import pandas as pd

from orion.storage.base import Storage
from orion.core.utils import flatten

from olympus.utils import get_storage as resolve_storage


def trials_as_dataframe(trials):
    dataframe_dict = []
    for t in trials:
        trial = dict(
            id=t.id,
        )

        trial.update(flatten.flatten(t.params))

        for result in t.results:
            trial[result.name] = result.value

        dataframe_dict.append(trial)

    return pd.DataFrame(dataframe_dict)


def to_html(obj, depth=0):
    if isinstance(obj, dict):
        return dict_to_html(obj, depth + 1)

    elif isinstance(obj, list):
        return list_to_html(obj, depth + 1)

    return obj


def dict_to_html(obj, depth=0):
    children = []
    for k, v in obj.items():
        children.append(
            html.Li([
                html.Span(k, className='pydict_key'),
                html.Span(to_html(v, depth + 1), className='pydict_value')
            ], className='pydict_item')
        )

    return html.Ul(children, className='pydict')


def list_to_html(obj, depth=0):
    return html.Ul([html.Li(to_html(i, depth + 1), className='pylist_value') for i in obj], className='pylist')


class ApplicationState:
    def __init__(self):
        self.storage = None
        self.experiments = None
        self.experiments_names = []
        self.trials = []
        self.trial_objective_name = ''
        self.current_exp = None

    def connect(self, storage_uri):
        storage_config = resolve_storage(storage_uri)
        storage_type = storage_config.pop('type')

        self.storage = Storage(of_type=storage_type, **storage_config)
        self.experiments = self.storage.fetch_experiments({})
        self.experiments_names = [e['name'] for e in self.experiments]
        self.experiments = {
            e['name']: e for e in self.experiments
        }

    def fetch_trials(self, exp_name):
        self.current_exp = exp_name
        self.trials = self.storage.fetch_trials(uid=self.experiments[exp_name]['_id'])

    def get_trial(self, id):
        return self.storage.get_trial(uid=id).to_dict()


state = ApplicationState()


def experiment_side_panel():
    return html.Div(
        className='small-col',
        children=[
            html.H4('Experiments'),
            dcc.RadioItems(
                id='experiment_name',
                options=[{'label': n, 'value': n} for n in state.experiments_names],
                value=None
            )
        ]
    )


def trials_side_panel():
    def format_objective(i, obj):
        state.trial_objective_name = obj.name
        return f'{i} ({obj.value:.4f})'

    items = dcc.RadioItems(
        id='trials_id',
        options=[{'label': format_objective(i, t.objective), 'value': t.id} for i, t in enumerate(state.trials)],
        value=None
    )

    return [
        html.H4(f'Trials ({state.trial_objective_name})'),
        items
    ]


def plot_button():
    return html.Div([
        html.Button('plot stuff', id='plot_button'),
    ])


def main(storage_uri):
    state.connect(storage_uri)

    app = dash.Dash(
        __name__,
        assets_folder=os.path.join(os.path.dirname(__file__), 'assets'),
        external_stylesheets=[
            'https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css'
        ]
    )

    app.config['suppress_callback_exceptions'] = True

    explorer = html.Div([
        # html.H3('Explorer'),
        html.Div(
            className='row container',
            children=[
                experiment_side_panel(),
                html.Div(id='trials_side_panel', className='small-col'),
                html.Div(id='exp-details', className='bigger-col'),
                html.Div(id='trial-details', className='bigger-col')
            ])])

    toolbox = html.Div(
        className='toolbox',
        children=[
            plot_button()
        ]
    )

    workspace = html.Div(
        id='workspace',
        children=[]
    )

    app.layout = html.Div(
        children=[
            toolbox,
            workspace,
            explorer
        ]
    )

    @app.callback(
        Output(component_id='trials_side_panel', component_property='children'),
        [Input(component_id='experiment_name', component_property='value')])
    def show_trials(exp_name):
        if exp_name is not None:
            state.fetch_trials(exp_name)
            return trials_side_panel()

    @app.callback(
        Output(component_id='exp-details', component_property='children'),
        [Input(component_id='experiment_name', component_property='value')])
    def get_experiment_details(exp_name):
        if exp_name is not None:
            data = state.experiments[exp_name]
            return [html.H4('Experiment: {}'.format(exp_name)), to_html(data)]

        return None

    @app.callback(
        Output(component_id='trial-details', component_property='children'),
        [Input(component_id='trials_id', component_property='value')])
    def get_trial_details(trial_id):

        def simplify(values):
            new_results = []
            for item in values:
                if isinstance(item['value'], float):
                    new_results.append('{}: {:.4f}'.format(item['name'], item['value']))
                else:
                    new_results.append('{}: {}'.format(item['name'], item['value']))
            return new_results

        if trial_id is not None:
            data = state.get_trial(trial_id)
            data['params'] = simplify(data.pop('params', []))
            data['results'] = simplify(data.pop('results', []))
            return [html.H4('Trial: {}'.format(trial_id)), to_html(data)]

        return None

    @app.callback(
        Output('workspace', component_property='children'),
        [Input('plot_button', 'n_clicks')])
    def plot_things(clicks):
        if clicks and state.current_exp:
            x = []
            y = []

            for t in state.trials:
                for r in t.results:
                    if r.name == 'epochs':
                        x.append(r.value)

                    if r.name == 'validation_accuracy':
                        y.append(r.value)

            fig = px.scatter(x, y)
            return dcc.Graph(figure=fig)

    app.run_server(debug=True)


# minimalist_hpo
if __name__ == '__main__':
    main('legacy:pickleddb:full_test.pkl')
