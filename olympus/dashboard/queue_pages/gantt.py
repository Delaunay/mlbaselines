from olympus.dashboard.queue_pages.inspect import InspectQueue
from olympus.dashboard.queue_pages.utilities import extract_work_messages
from olympus.dashboard.plots.work_distribution import plot_gantt_plotly, prepare_gantt_array
import olympus.dashboard.elements as html

from msgqueue.backends import new_monitor


def make_plot(client, queue, namespace):
    if isinstance(client, dict):
        client = new_monitor(**client)

    messages = client.messages(queue, namespace)
    work_items, worker_count = extract_work_messages(messages)

    if len(work_items) == 0:
        return None

    jobs, annotations, resources = prepare_gantt_array(work_items, worker_count)
    return plot_gantt_plotly(jobs, annotations, resources)


class GanttQueue(InspectQueue):
    base_path = 'queue/gantt'

    def __init__(self, client):
        super(GanttQueue, self).__init__(client)
        self.title = 'Work Distribution'
        self.plot_id = 'plotly_id'

    def show_queue(self, queue, namespace):
        return html.async_plotly_plot(
            make_plot,
            self.client.state_dict(),
            queue,
            namespace)
