from argparse import ArgumentParser
import os

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go

import pandas as pd

from orion.storage.base import Storage
from orion.core.utils import flatten

from olympus.utils import get_storage as resolve_storage


state: None


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
        self.extracted_data = None
        self.columns = []
        self.args = None

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
        self.extract_data()

    def get_trial(self, id):
        return self.storage.get_trial(uid=id).to_dict()

    def extract_data(self):
        results = []

        for trial in self.trials:
            trial_info = {'id': trial.id}

            for k, v in trial.params.items():
                if k in trial_info:
                    print(f'warning, overriding {k}')

                trial_info[k] = v

            for result in trial.results:
                if result.name in trial_info:
                    print(f'warning, overriding {result.name}')

                trial_info[result.name] = result.value

            results.append(trial_info)

        self.extracted_data = pd.DataFrame(results)
        self.columns = list(self.extracted_data.columns)


def experiment_side_panel():
    return html.Div(
        className='col-2',
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
        if obj is None:
            return str(i)

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


def graph_dials():
    return html.Div(id='graph_dials', style={'display': 'inline'})


def prettify_name(name):
    return name.replace('_', ' ').capitalize()


def make_dials():
    choices = [{'label': prettify_name(i), 'value': i} for i in state.columns]
    options = [{'label': i, 'value': i} for i in ['linear', 'log']]
    return html.Div(
        className='form-group row',
        children=[
            html.Div(
                className='col-sm-4',
                children=[
                    html.Div(
                        dcc.Dropdown(
                            className='dropdown',
                            placeholder='x axis',
                            id='xaxis-column',
                            options=choices,

                        ),
                        style={'display': 'inline-block', 'width': '50%', 'vertical-align': 'middle'}),
                    dcc.RadioItems(
                        id='xaxis-type',
                        options=options,
                        value='linear',
                        style={'display': 'inline-block', 'padding': '5px'}
                    )]),
            html.Div(
                className='col-sm-4',
                children=[
                    html.Div(
                        dcc.Dropdown(
                            className='dropdown',
                            placeholder='y axis',
                            id='yaxis-column',
                            options=choices,
                            style={'vertical-align': 'middle'}
                        ),
                        style={'display': 'inline-block', 'width': '50%', 'vertical-align': 'middle'}),
                    dcc.RadioItems(
                        id='yaxis-type',
                        options=options,
                        value='linear',
                        style={'display': 'inline-block', 'padding': '5px'}
                    )]),
        ])


def main(args=None):
    global state

    parser = ArgumentParser()
    parser.add_argument('--storage-uri', type=str, default='legacy:pickleddb:full_test.pkl')

    args = parser.parse_args(args)

    state = ApplicationState()
    state.args = args
    state.connect(args.storage_uri)

    app = dash.Dash(
        __name__,
        assets_folder=os.path.join(os.path.dirname(__file__), 'assets'),
        external_stylesheets=[
            'https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css'
        ]
    )

    app.config['suppress_callback_exceptions'] = True

    explorer = html.Div(
        children=[
            # html.H3('Explorer'),
            html.Div(
                className='row',
                children=[
                    experiment_side_panel(),
                    html.Div(id='trials_side_panel', className='col-2'),
                    html.Div(id='exp-details', className='col-4'),
                    html.Div(id='trial-details', className='col-4')
                ])])

    toolbox = html.Div(
        className='toolbox',
        children=[
            graph_dials()
        ]
    )

    workspace = html.Div(
        id='workspace',
        className='workspace_class',
        children=[
            dcc.Graph(id='main-graph')
        ]
    )

    app.layout = html.Div(
        className='container-fluid',
        children=[
            toolbox,
            workspace,
            explorer
        ]
    )

    # Fetch the details of a specific experiment
    @app.callback(
        Output(component_id='exp-details', component_property='children'),
        [Input(component_id='experiment_name', component_property='value')])
    def get_experiment_details(exp_name):
        if exp_name is not None:
            data = state.experiments[exp_name]
            return [html.H4('Experiment: {}'.format(exp_name)), to_html(data)]

        return None

    # show all the trials of a given experiment
    @app.callback(
        [Output(component_id='trials_side_panel', component_property='children'),
         Output(component_id='graph_dials', component_property='children')],
        [Input(component_id='experiment_name', component_property='value')])
    def show_trials(exp_name):
        if exp_name is not None:
            state.fetch_trials(exp_name)

            return trials_side_panel(), make_dials()

        return None, None

    # Get the details of a specific trial
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
        Output('main-graph', 'figure'),
        [Input('xaxis-column', 'value'),
         Input('yaxis-column', 'value'),
         Input('xaxis-type', 'value'),
         Input('yaxis-type', 'value')])
    def update_graph(xaxis_col, yaxis_col, xaxis_type, yaxis_type):
        if xaxis_col is None or yaxis_col is None:
            return {}

        xaxis_type = xaxis_type.lower()
        yaxis_type = yaxis_type.lower()

        x_data = state.extracted_data[xaxis_col]
        y_data = state.extracted_data[yaxis_col]

        return {
            'data': [go.Scatter(
                x=x_data,
                y=y_data,
                mode='markers',
                marker={
                    'size': 15,
                    'opacity': 0.5,
                    'line': {'width': 0.5, 'color': 'white'}
                }
            )],
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
    main()

