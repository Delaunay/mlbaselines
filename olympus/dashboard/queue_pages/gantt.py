import rpcjs.elements as html

from olympus.dashboard.queue_pages.inspect import InspectQueue
from olympus.dashboard.queue_pages.utilities import extract_work_messages
from olympus.dashboard.plots.work_distribution import plot_gantt_plotly, prepare_gantt_array

from msgqueue.backends import new_monitor


def make_plot(client, queue, namespace, color, delimiter, start_date):
    if isinstance(client, dict):
        client = new_monitor(**client)

    if delimiter is None:
        messages = client.messages(queue, namespace)
    else:
        client = client.aggregate_monitor()
        messages = client.messages(queue, namespace, delimiter=delimiter)

    work_items, worker_count = extract_work_messages(messages)

    if len(work_items) == 0:
        return None

    jobs, annotations, resources = prepare_gantt_array(work_items, worker_count)
    return plot_gantt_plotly(jobs, color, annotations, resources)


class GanttQueue(InspectQueue):
    base_path = 'queue/gantt'

    def routes(self):
        return [
            f'/{self.base_path}',
            f'/{self.base_path}/<string:queue>',
            f'/{self.base_path}/<string:queue>/<string:namespace>/<string:start_date>',
        ]

    def __init__(self, client):
        super(GanttQueue, self).__init__(client)
        self.title = 'Work Distribution'
        self.plot_id = 'plotly_id'

    def main(self, queue=None, namespace=None, start_date=None):
        if queue is None:
            return self.list_queues()

        if namespace is None:
            return self.list_namespaces(queue)

        return self.show_queue(queue, namespace, start_date)

    def show_queue(self, queue, namespace, delimiter=None, color='epoch', start_date=None):
        return html.async_plotly_plot(
            make_plot,
            self.client.state_dict(),
            queue,
            namespace,
            color,
            delimiter,
            start_date)
