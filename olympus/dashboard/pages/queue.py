import dash_core_components as dcc
import dash_html_components as html

import plotly.figure_factory as ff
import plotly.graph_objects as go

import altair as alt

from olympus.dashboard.analysis.hpfanova import FANOVA
from olympus.dashboard.render import MessagesRender, AgentsRender
from olympus.dashboard.pages.page import Page
from olympus.dashboard.base import get_kv
from olympus.dashboard.graph import Altair, PyPlot
from olympus.dashboard.viz import Matrix

from olympus.hpo.parallel import WORK_ITEM, HPO_ITEM, WORKER_LEFT, WORKER_JOIN, RESULT_ITEM, SHUTDOWN


from msgqueue.backends.queue import QueueMonitor
from msgqueue.worker import RESULT_QUEUE, WORK_QUEUE


mappings = {
    WORK_ITEM: 'trial',
    HPO_ITEM: 'hpo',
    WORKER_LEFT: 'worker_left',
    WORKER_JOIN: 'worker_join',
    RESULT_ITEM: 'result',
    SHUTDOWN: 'shutdown'
}


namespaces = None
queue_names = {}


def get_namespaces(client):
    global namespaces
    if not namespaces:
        namespaces = list(set(client.namespaces()))
    return namespaces


def get_queues(client, namespace):
    global queue_names
    names = queue_names.get(namespace)
    if names is None:
        names = client.queues()
        queue_names[namespace] = names

    return names


def list_namespaces(client, page_name, link):
    namespaces_ = get_namespaces(client)
    comb = set()

    items = []
    for namespace in namespaces_:
        queues = []

        for name in get_queues(client, namespace):
            if (namespace, name) not in comb:
                comb.add((namespace, name))
            else:
                continue

            queues.append(
                html.Li(dcc.Link(f'{name}/{namespace}', href=f'/queue/{link}/{name}/{namespace}')))

        items.append(html.Li([f'Queues for {namespace}', html.Ul(queues)]))

    return html.Div([
        html.H4(f'Namespaces for {page_name}'),
        html.Ul(items)])


class QueuePage(Page):
    @staticmethod
    def route():
        return '/queue/raw/?(?P<queue_name>[a-zA-Z0-9]*)/?(?P<namespace>[a-zA-Z0-9_\\-]*)'

    def __init__(self, client: QueueMonitor):
        self.client = client

    def render(self, app, queue_name, namespace):
        if not namespace:
            return list_namespaces(self.client, 'raw', 'raw')

        app.title = f'Raw {namespace}/{queue_name}'
        return self.pending_messages(app, queue_name, namespace)

    def pending_messages(self, app, name, namespace):
        messages = self.client.messages(name, namespace)
        print(messages, name, namespace)
        return MessagesRender(messages).render(app)


class ResultPage(Page):
    @staticmethod
    def route():
        return '/queue/result/?(?P<queue_name>[a-zA-Z0-9]*)/?(?P<namespace>[a-zA-Z0-9]*)'

    def __init__(self, client: QueueMonitor):
        self.client = client

    def render(self, app, namespace, queue_name):
        if not namespace:
            return list_namespaces(self.client, 'result', 'result')

        app.title = f'Results {queue_name}/{namespace}'
        return self.show_results(namespace, queue_name)

    def extract_messages(self, messages):
        results = []
        for m in messages:
            if m.mtype == RESULT_ITEM:
                params, result = m.message
                results.append(dict(
                    trial=params['uid'],
                    epoch=params['epoch'],
                    objective=result
                ))

        return results

    def show_results(self, namespace, queue_name):
        messages = self.client.messages(queue_name, namespace)

        if get_kv('is_dark'):
            alt.themes.enable('dark')

        results = alt.Data(values=self.extract_messages(messages))

        line = alt.Chart(results).mark_line().encode(
            x=alt.X('epoch:Q'),
            y='mean(objective):Q'
        )
        band = alt.Chart(results).mark_errorband(extent='ci').encode(
            x=alt.X('epoch:Q'),
            y=alt.Y('objective:Q', title='objective'),
        )

        graph = (band + line).configure_view(height=500, width=1000)

        return Altair(graph).render(None)


class GanttPage(Page):
    @staticmethod
    def route():
        return '/queue/gantt/?(?P<queue_name>[a-zA-Z0-9]*)/?(?P<namespace>[a-zA-Z0-9]*)'

    def __init__(self, client: QueueMonitor):
        self.client = client

    def render(self, app, queue_name, namespace):
        if not namespace:
            return list_namespaces(self.client, 'gantt', 'gantt')

        app.title = f'Gantt {namespace}/{queue_name}'
        return self.gantt(namespace, queue_name)

    def gantt(self, namespace, name):
        messages = self.client.messages(name, namespace)
        worker_count = len(self.client.agents())
        jobs, annotations = self.prepare_gantt_data(messages, worker_count)

        if not jobs:
            return 'No work item found'

        fig = ff.create_gantt(
            jobs, title='Work Schedule', index_col='Resource', showgrid_x=True, showgrid_y=True,
            show_colorbar=True, group_tasks=True, bar_width=0.4)

        fig['layout']['annotations'] = annotations

        if get_kv('is_dark'):
            fig.update_layout(template='plotly_dark')

        for data in fig['data']:
            data['marker']['symbol'] = 'line-ns'
            data['marker']['size'] = 20
            data['marker']['opacity'] = 1
            data['marker']['line'] = {
                'width': 1
            }

        return dcc.Graph(
            figure=fig
        )

    def extract_messages(self, messages):
        worker_count = 0
        worker_left = 0
        worker_shutdown = 0

        work_items = []
        for m in messages:
            if not m.read_time or not m.actioned_time:
                continue

            if m.mtype == WORKER_JOIN:
                worker_count += 1

            elif m.mtype == WORKER_LEFT:
                worker_left += 1

            elif m.mtype == HPO_ITEM or m.mtype == WORK_ITEM:
                work_items.append(m)

            elif m.mtype == RESULT_ITEM:
                pass

            elif m.mtype == SHUTDOWN:
                worker_shutdown += 1
            else:
                print(m.mtype, 'ignored')

        work_items.sort(key=lambda m: m.read_time)
        # This is unreliable
        # worker_count = max(worker_count, worker_left, worker_shutdown)
        return work_items

    def prepare_gantt_data(self, messages, worker_count):
        work_items = self.extract_messages(messages)

        workers = [None for _ in range(worker_count + 1)]

        def find_free_worker(start_time, end_time):
            for i, worker in enumerate(workers):
                if worker is None or worker < start_time:
                    workers[i] = end_time
                    return i

            return 'U'

        jobs = []
        annotations = []
        for w in work_items:
            worker_id = find_free_worker(w.read_time, w.actioned_time)

            resource = mappings.get(w.mtype)
            epoch = dict(w.message.get('kwargs', [])).get('epoch', None)
            if epoch is not None:
                resource = f'{resource} ({epoch})'

            task = f'worker-{worker_id}'
            jobs.append(dict(
                Task=task,
                Start=w.read_time,
                Finish=w.actioned_time,
                Resource=resource))

            # annotations.append(dict(
            #     x=w.read_time + (w.actioned_time - w.read_time) / 2,
            #     y=worker_id,
            #     text=str(w.uid)[:4],
            #     showarrow=True))

        return jobs, annotations


class StatusPage(Page):
    @staticmethod
    def route():
        return '/queue/status/?(?P<queue_name>[a-zA-Z0-9]*)/?(?P<namespace>[a-zA-Z0-9]*)'

    def __init__(self, client: QueueMonitor):
        self.client = client

    def render(self, app, namespace, queue_name):
        if not namespace:
            return list_namespaces(self.client, 'status', 'status')

        app.title = f'Status {namespace}/{queue_name}'
        return self.show_status(namespace, queue_name)

    def get_unique_agents(self, namespace):
        """Agent are not unique every time one is restarted a new entry is added
        The list is already sorted so we know we get the last agent for each name
        """
        agents = self.client.agents()
        unique_agents = {}
        for a in agents:
            unique_agents[a.agent] = a
        return list(unique_agents.values())

    def show_status(self, namespace, name):
        unread_messages = self.client.unread_messages(name, namespace)
        unactioned      = self.client.unactioned_messages(name, namespace)
        finished        = self.client.actioned_count(name, namespace)
        lost            = self.client.lost_messages(name, namespace)

        agents = self.get_unique_agents(namespace)

        data = {
            'pending': len(unread_messages),
            'running': len(unactioned),
            'finished': finished,
            'lost': len(lost)
        }

        fig = go.Figure(data=[
            go.Pie(labels=tuple(data.keys()), values=tuple(data.values()))])
        fig.update_traces(hoverinfo='label+percent', textinfo='value')

        if get_kv('is_dark'):
            fig.update_layout(template='plotly_dark')

        return html.Div([
            html.H4(f'Status of {namespace}/{name}'),
            html.Div(className='row', children=[
                html.Div(className='col-4', children=[
                    html.H5(f'Tasks'),
                    dcc.Graph(figure=fig)]),
                html.Div(className='col', children=[
                    html.H5(f'Agents'),
                    AgentsRender(agents, namespace).render(None)]),
            ]),
            html.Div(className='row', children=[
                html.Div(className='col', children=[
                    html.H5(f'Unread (limit: 10)'),
                    MessagesRender(unread_messages[:10]).render(None)
                ]),
                html.Div(className='col', children=[
                    html.H5(f'Running (limit: 10)'),
                    MessagesRender(unactioned[:10]).render(None)
                ]),
            ])
        ])


class LogPage(Page):
    @staticmethod
    def route():
        return '/queue/logs/?(?P<namespace>[a-zA-Z0-9]*)/?(?P<agent_id>[a-zA-Z0-9]*)'

    def __init__(self, client: QueueMonitor):
        self.client = client

    def render(self, app, namespace, agent_id):
        if not namespace or not agent_id:
            return

        app.title = f'Log {namespace}/{agent_id}'
        return self.show_logs(namespace, agent_id)

    def show_logs(self, namespace, agent):
        data = self.client.log(agent)
        return html.Div([
            html.H4(f'Logs of {namespace}/{agent}'),
            html.Pre(data)
        ])


class SpaceExploration(Page):
    @staticmethod
    def route():
        return '/queue/space_exp/?(?P<namespace>[a-zA-Z0-9]*)/?(?P<queue>[a-zA-Z0-9]*)'

    def __init__(self, client: QueueMonitor):
        self.client = client

    def render(self, app, namespace, queue):
        if not namespace or not queue:
            return

        app.title = f'Space {namespace}/{queue}'
        return self.show_space_exploration(namespace, queue)

    def extract_messages(self, messages, unique=False):
        import copy

        if not unique:
            results = []
        else:
            results = {}

        columns = set()

        # Keep the most recent trial
        for m in messages:
            if m.mtype == WORK_ITEM:
                uid = m.message['kwargs']['uid']

                params = copy.deepcopy(m.message['kwargs'])
                params.pop('uid')
                columns.update(params.keys())

                if not unique:
                    results.append(params)
                else:
                    old = results.get(uid)
                    if old is None:
                        results[uid] = params
                    elif old['epoch'] < params['epoch']:
                        results[uid] = params

        columns.discard('epoch')

        if not unique:
            values = list(sorted(results, key=lambda p: p['epoch']))
        else:
            values = list(sorted(results.values(), key=lambda p: p['epoch']))

        return values, list(columns)

    def plotly_scatter_matrix(self, data, columns):
        # Looks ugly
        import pandas as pd

        df = pd.DataFrame(data)
        index_vals = df['epoch'].astype('category').cat.codes

        fig = go.Figure(data=go.Splom(
            showlowerhalf=False,
            diagonal_visible=False,
            text=df['epoch'],
            dimensions=[
                dict(label=col, values=df[col]) for col in columns],
            marker=dict(
                color=index_vals,
                showscale=False,
                line_color='white',
                line_width=0.5)))

        if get_kv('is_dark'):
            fig.update_layout(template='plotly_dark')

        fig.update_layout(
            showlegend=True,
            width=600,
            height=600)

        return dcc.Graph(figure=fig)

    def altair_scatter_repeat(self, data, columns):
        # Looks good but has duplicated charts
        space = alt.Data(values=data)

        chart = alt.Chart(space).mark_circle().encode(
            alt.X(alt.repeat("column"), type='quantitative'),
            alt.Y(alt.repeat("row"), type='quantitative'),
            color='epoch:N'
        ).properties(
            width=200,
            height=200
        ).repeat(
            row=columns,
            column=columns
        ).interactive()

        return Altair(chart).render()

    def altair_scatter_matrix(self, data, columns):
        space = alt.Data(values=data)

        base = alt.Chart().properties(
            width=120,
            height=120
        )

        def scatter_plot(row, col):
            """Standard Scatter plot"""
            return base.mark_circle(size=5).encode(
                alt.X(row, type='quantitative'),
                alt.Y(col, type='quantitative'),
                color='epoch:N'
            ).interactive()

        def density_plot(row):
            """Estimate the density function using KDE"""
            return base.transform_density(
                row,
                as_=[row, 'density']
            ).mark_line().encode(
                x=f'{row}:Q',
                y='density:Q'
            )

        def histogram_plot(row):
            """Show density as an histogram"""
            return base.mark_bar().encode(
                alt.X(row, type='quantitative', bin=True),
                y='count()'
            )

        chart = (Matrix(space)
                 .fields(*columns)
                 # .upper(scatter_plot)
                 .diag(histogram_plot)
                 .lower(scatter_plot)).render()

        return Altair(chart).render()

    def show_space_exploration(self, namespace, queue):
        messages = self.client.messages(namespace, queue)

        if get_kv('is_dark'):
            alt.themes.enable('dark')

        data, columns = self.extract_messages(messages)

        # graph = self.altair_scatter_repeat(data, columns)
        graph = self.altair_scatter_matrix(data, columns)

        return html.Div([
            html.H4(f'Space Exploration of {namespace}/{queue}'),
            graph
        ])


class FANOVAPage(Page):
    @staticmethod
    def route():
        return '/queue/fanova/?(?P<queue_name>[a-zA-Z0-9]*)/?(?P<namespace>[a-zA-Z0-9]*)'

    def __init__(self, client: QueueMonitor):
        self.client = client

    def render(self, app, namespace, queue_name):
        if not namespace:
            return list_namespaces(self.client, 'fANOVA', 'fANOVA')

        app.title = f'fANOVA {namespace}/{queue_name}'
        return self.show_hyperparameter_importance(app, namespace, queue_name)

    def extract_messages(self, messages):
        results = {}
        columns = set()

        # Keep the most recent trial
        for m in messages:
            if m.mtype == RESULT_ITEM:
                uid, params, objective = m.message

                params = dict(params)
                params['objective'] = objective

                columns.update(params.keys())
                results[uid] = params

        return list(results.values()), list(columns)

    def show_marginals(self, fanova):
        data = fanova.compute_marginals()
        marginals = alt.Data(values=data)

        base = alt.Chart(marginals).encode(
            alt.X('value', type='quantitative'),
            alt.Y('objective', type='quantitative'),
            yError='std:Q',
            color='name:N'
        ).properties(
            width=200,
            height=200
        )

        chart = (base.mark_errorband() + base.mark_line())\
            .facet(column='name:N').interactive()

        return Altair(chart).render()

    def importance_heatmap(self, fanova, columns):
        fig = go.Figure(data=go.Heatmap(
            z=fanova.importance, x=columns, y=list(reversed(columns))))

        fig_std = go.Figure(data=go.Heatmap(
            z=fanova.importance_std, x=columns, y=list(reversed(columns))))

        if get_kv('is_dark'):
            fig.update_layout(template='plotly_dark')
            fig_std.update_layout(template='plotly_dark')

        return html.Div(className='row', children=[
            html.Div(className='col', children=[
                html.H4('Importance'),
                dcc.Graph(figure=fig)]),
            html.Div(className='col', children=[
                html.H4('Importance std'),
                dcc.Graph(figure=fig_std)])])

    def importance_heatmap_altair(self, fanova):
        data = alt.Data(values=fanova.importance_long)

        base = alt.Chart(data).mark_rect().encode(
            x='row:O',
            y='col:O'
        ).properties(
            width=200,
            height=200
        )

        chart = alt.concat(
            base.encode(color='importance:Q'),
            base.encode(color='std:Q')
        ).resolve_scale(
            color='independent'
        )

        return html.Div([html.H4('Importance'), Altair(chart).render(None)], style=dict(height=300))

    def show_hyperparameter_importance(self, app, namespace, queue):
        messages = self.client.messages(namespace, [WORK_QUEUE, RESULT_QUEUE])

        if get_kv('is_dark'):
            alt.themes.enable('dark')

        import pandas as pd

        data, columns = self.extract_messages(messages)

        # Make a list of all hyper parameters
        columns = set(columns)
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
                'b': 'uniform(0, 1)',
                'c': 'uniform(0, 1)',
                'lr': 'uniform(0, 1)'
            })

        return html.Div([
            # self.importance_heatmap(fanova, columns),
            self.importance_heatmap_altair(fanova),
            html.H4('Marginals'),
            self.show_marginals(fanova)
        ], style=dict(height='100$'))
