from collections import defaultdict

from olympus.dashboard.queue_pages.inspect import InspectQueue
from olympus.dashboard.plots.work_status import work_status
import olympus.dashboard.elements as html


class StatusQueue(InspectQueue):
    base_path = 'queue/status'

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

    def main(self, queue=None, namespace=None):
        if queue is None:
            return self.list_queues()

        if namespace is None:
            return self.show_overview(queue)

        return self.show_queue(queue, namespace)

    @staticmethod
    def insert(aggregated, metrics):
        for metric in metrics:
            namespace = metric.pop('_id')
            aggregated[namespace][namespace] = namespace
            for k, v in metric.items():
                aggregated[namespace][k] = v

        return aggregated

    def altair_overview(self, data):
        altair_data = []
        agents = []
        for namespace, statuses in data.items():
            altair_data.append(dict(experiment=namespace, message='pending', count=statuses['unread']))
            altair_data.append(dict(experiment=namespace, message='in-progress', count=statuses['unactioned']))
            altair_data.append(dict(experiment=namespace, message='finished', count=statuses['actioned']))
            altair_data.append(dict(experiment=namespace, message='lost', count=statuses['lost']))
            altair_data.append(dict(experiment=namespace, message='failed', count=statuses['failed']))
            agents.append(dict(experiment=namespace, message='agents', count=statuses['agent']))

        import altair as alt
        alt.themes.enable('dark')

        data = alt.Data(values=altair_data)
        chart = alt.Chart(data, title='Message status per experiment').mark_bar().encode(
            x=alt.X('count:Q', stack='normalize'),
            y='experiment:N',
            color='message:N'
        )

        data = alt.Data(values=agents)
        agent_chart = alt.Chart(data, title='Agent per experiment').mark_bar().encode(
            x=alt.X('count:Q'),
            y='experiment:N',
            color='message:N')

        # return chart
        return alt.vconcat(chart, agent_chart)

    def list_experiment(self, queue, experiments, data):

        def show_expriment(n):
            status = data.get(n, {})
            link = html.link(n, ref=f'/{self.base_path}/{queue}/{n}')
            row = f"""
                <tbody>
                    <tr>
                        <td>{link}</td>
                        <td>{status.get("lost", 0)}</td>
                        <td>{status.get("failed", 0)}</td>
                        <td>{status.get("agent", 0)}</td>
                    </tr>
                </tbody>"""
            return row

        rows = ''.join([show_expriment(name) for name in filter(lambda e: e is not None, experiments)])

        return html.div(
            html.header(f'Details {queue}', level=4),
            f"""
            <table class="table table-hover table-striped table-sm">
                <thead>
                    <tr>
                        <th>Experiment</th>
                        <th>Lost</th>
                        <th>Failed</th>
                        <th>Agents</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>""")

    def show_namespace(self, name, status, agents):
        fig = work_status(status)
        fig.update_layout(template='plotly_dark')

        return html.div(
            html.header(f'{name} {agents}', level=5),
            html.div_col(html.plotly_plot(fig), size=4))

    def show_overview(self, queue):
        # Try to show an overview of the entire system
        if self.aggregate is None:
            try:
                self.aggregate = self.client.aggregate_monitor()
            except:
                return self.list_namespaces(queue)

        data = defaultdict(lambda: dict(unread=0, unactioned=0, actioned=0, lost=0, failed=0, agent=0))

        self.insert(data, self.aggregate.unread_count(queue))
        self.insert(data, self.aggregate.unactioned_count(queue))
        self.insert(data, self.aggregate.actioned_count(queue))
        self.insert(data, self.aggregate.lost_count(queue))
        self.insert(data, self.aggregate.failed_count(queue))
        self.insert(data, self.aggregate.agent_count())

        chart = self.altair_overview(data)
        return html.div(
            html.header('Experiments', level=3),
            html.div_row(
                html.div_col(html.altair_plot(chart), style='height:500px;', size=5),
                html.div_col(self.list_experiment(queue, data.keys(), data))
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
            html.header(f'Status of {queue}/{namespace}', level=3),
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
