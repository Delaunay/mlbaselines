from olympus.dashboard.queue_pages.inspect import InspectQueue
from olympus.dashboard.queue_pages.utilities import extract_work_messages
from olympus.dashboard.plots.work_distribution import plot_gantt_plotly, prepare_gantt_array
from olympus.dashboard.dash import set_attribute, async_call
import olympus.dashboard.elements as html

from msgqueue.backends import new_monitor

from olympus.hpo.parallel import make_remote_call


def make_plot(client, queue, namespace, plot_id):
    if isinstance(client, dict):
        client = new_monitor(**client)

    messages = client.messages(queue, namespace)
    work_items, worker_count = extract_work_messages(messages)

    jobs, annotations, resources = prepare_gantt_array(work_items, worker_count)
    fig = plot_gantt_plotly(jobs, annotations, resources)

    html_plot = html.plotly_plot(fig, full_html=True)

    # Tell the main server to execute this final function
    return make_remote_call(
        set_attribute, plot_id, 'srcdoc', html_plot)


class GanttQueue(InspectQueue):
    base_path = 'queue/gantt'

    def __init__(self, client):
        super(GanttQueue, self).__init__(client)
        self.title = 'Work Distribution'
        self.plot_id = 'plotly_id'

    def show_queue(self, queue, namespace):
        # This might take some time so reply first and start the work later
        async_call(
            make_plot,
            self.client.state_dict(),
            queue,
            namespace,
            self.plot_id)

        return html.iframe(html.iframe_spinner(), id=self.plot_id)
