from olympus.dashboard.queue_pages.inspect import InspectQueue
from olympus.dashboard.queue_pages.utilities import extract_work_messages
from olympus.dashboard.plots.work_distribution import plot_gantt_plotly, prepare_gantt_array
from olympus.dashboard.elements import plotly_plot


class GanttQueue(InspectQueue):
    base_path = 'queue/gantt'

    def __init__(self, client):
        super(GanttQueue, self).__init__(client)
        self.title = 'Work Distribution'

    def show_queue(self, queue, namespace):
        messages = self.client.messages(queue, namespace)
        work_items, worker_count = extract_work_messages(messages)
        jobs, annotations = prepare_gantt_array(work_items, worker_count)
        fig = plot_gantt_plotly(jobs, annotations)
        return plotly_plot(fig)
