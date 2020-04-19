from olympus.dashboard.queue_pages.inspect import InspectQueue
from olympus.dashboard.plots.work_status import work_status
import olympus.dashboard.elements as html


class StatusQueue(InspectQueue):
    base_path = 'queue/status'

    def __init__(self, client):
        super(StatusQueue, self).__init__(client)
        self.title = 'Queue Status'

    def get_unique_agents(self, queue, namespace):
        """Agent are not unique every time one is restarted a new entry is added
        The list is already sorted so we know we get the last agent for each name
        """
        agents = self.client.agents()
        unique_agents = {}
        for a in agents:
            unique_agents[a.agent] = a
        return list(unique_agents.values())

    def show_queue(self, queue, namespace):
        # unread_messages = self.client.unread_messages(queue, namespace)
        # unactioned = self.client.unactioned_messages(queue, namespace)

        unread = self.client.unread_count(queue, namespace)
        unactioned = self.client.unactioned_count(queue, namespace)
        finished = self.client.actioned_count(queue, namespace)

        lost = self.client.lost_messages(queue, namespace)
        failed = self.client.failed_messages(queue, namespace)

        # agents = self.get_unique_agents(queue, namespace)

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
            html.header(f'Status of {queue}/{namespace}'),
            html.div_row(
                html.div_col(
                    html.header('Tasks', level=5),
                    html.plotly_plot(fig), size=4),
                html.div_col(
                    html.header('Agents', level=5),
                    # TODO
                )
            ),
            html.div_row(
                html.div_col(
                    html.header('Lost', level=5),
                    html.show_messages(lost[:10])),
                html.div_col(
                    html.header('Failed', level=5),
                    html.show_messages(failed[:10])))
            )
