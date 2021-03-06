import rpcjs.elements as html

from collections import defaultdict

from olympus.dashboard.queue_pages.inspect import InspectQueue
from olympus.dashboard.plots.work_status import work_status, aggregate_overview_altair, prepare_overview_altair


def list_experiment_stats(base_path, queue, experiments, data, nolink):
    def show_expriment(n):
        status = data.get(n, {})
        if not nolink:
            link = html.link(n, ref=f'/{base_path}/{queue}/{n}')
        else:
            link = n

        row = f"""
            <tbody>
                <tr>
                    <td>{link}</td>
                    <td>{status.get("lost", 0)}</td>
                    <td>{status.get("failed", 0)}</td>
                    <td>{status.get("agent", 0)}</td>
                    <td>{status.get("runtime_actioned", 0):0.2f}</td>
                </tr>
            </tbody>"""
        return row

    rows = ''.join([show_expriment(name) for name in filter(lambda e: e is not None, experiments)])
    return html.div(
        f"""
        <table class="table table-hover table-striped table-sm">
            <thead>
                <tr>
                    <th>Experiment</th>
                    <th>Lost</th>
                    <th>Failed</th>
                    <th>Agents</th>
                    <th>Runtime</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>""")


def insert(aggregated, metrics, key):
    for metric in metrics:
        namespace = metric.pop('_id')
        aggregated[namespace][namespace] = namespace

        for k, v in metric.items():
            if k == 'runtime':
                aggregated[namespace][f'{k}_{key}'] = v

            elif v is not None:
                aggregated[namespace][k] = v

    return aggregated


class StatusQueue(InspectQueue):
    base_path = 'queue/status'

    def routes(self):
        return [
            f'/{self.base_path}',
            f'/{self.base_path}/<string:queue>',
            f'/{self.base_path}/<string:queue>/<string:namespace>',
        ]

    def __init__(self, client):
        super(StatusQueue, self).__init__(client)
        self.title = 'Queue Status'
        self.aggregate = None

    def get_unique_agents(self, queue, namespace):
        """Agent are not unique every time one is restarted a new entry is added
        The list is already sorted so we know we get the last agent for each name
        """
        agents = self.client.agents(namespace)
        unique_agents = {}
        for a in agents:
            unique_agents[a.agent] = a
        return list(unique_agents.values())

    def main(self, queue=None, namespace=None, group=None, delimiter=None):
        if queue is None:
            return self.list_queues()

        if namespace is None:
            return self.show_overview(queue, group, delimiter)

        return self.show_queue(queue, namespace)

    def show_namespace(self, name, status, agents):
        fig = work_status(status)
        fig.update_layout(template='plotly_dark')

        return html.div(
            html.header(f'{name} {agents}', level=5),
            html.div_col(html.plotly_plot(fig), size=4))

    def show_overview(self, queue, group=None, delimiter=None):
        # Try to show an overview of the entire system
        if self.aggregate is None:
            try:
                self.aggregate = self.client.aggregate_monitor()
            except Exception as  e:
                print(e)
                return self.list_namespaces(queue)

        data = defaultdict(lambda: dict(unread=0, unactioned=0, actioned=0, lost=0, failed=0, agent=0, runtime=0))
        insert(data, self.aggregate.unread_count(queue, group, delimiter=delimiter), 'unread')
        insert(data, self.aggregate.unactioned_count(queue, group, delimiter=delimiter), 'unactioned')
        insert(data, self.aggregate.actioned_count(queue, group, delimiter=delimiter), 'actioned')
        insert(data, self.aggregate.lost_count(queue, group, delimiter=delimiter), 'lost')
        insert(data, self.aggregate.failed_count(queue, group, delimiter=delimiter), 'failed')

        for group, info in data.items():
            print(group, info)

        name = 'experiment'
        if delimiter is not None:
            name = 'study'

        status, agents = prepare_overview_altair(data)
        chart = aggregate_overview_altair(status, name)

        return html.div(
            html.header('Status', level=3),
            html.div_row(
                html.div_col(html.altair_plot(chart), style='height:100vh;', size=5),
                html.div_col(list_experiment_stats(self.base_path, queue, data.keys(), data, nolink=delimiter is not None))
            )
        )

    def show_queue(self, queue, namespace):
        # unread_messages = self.client.unread_messages(queue, namespace)
        # unactioned = self.client.unactioned_messages(queue, namespace)

        unread = self.client.unread_count(queue, namespace)
        unactioned = self.client.unactioned_count(queue, namespace)
        finished = self.client.actioned_count(queue, namespace)

        lost = self.client.lost_messages(queue, namespace)
        failed = self.client.failed_messages(queue, namespace)

        agents = self.get_unique_agents(queue, namespace)

        data = {
            'pending': unread,
            'running': unactioned,
            'finished': finished,
            'lost': len(lost),
            'failed': len(failed)
        }

        fig = work_status(data)
        fig.update_layout(template='plotly_dark')

        return html.div(
            html.header(f'Status', level=3),
            html.div_row(
                html.div_col(
                    html.header('Tasks', level=5),
                    html.plotly_plot(fig), size=4),
                html.div_col(
                    html.header('Agents', level=5),
                    html.show_agent(agents))),
            html.div_row(
                html.div_col(
                    html.header('Lost', level=5),
                    html.show_messages(lost[:10])),
                html.div_col(
                    html.header('Failed', level=5),
                    html.show_messages(failed[:10]))))
