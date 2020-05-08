import rpcjs.elements as html

from olympus.dashboard.queue_pages.inspect import InspectQueue
from olympus.dashboard.queue_pages.utilities import objective_array
from olympus.dashboard.plots.training_curve import plot_mean_objective_altair


class ResultQueue(InspectQueue):
    base_path = 'queue/result'

    def __init__(self, client):
        super(ResultQueue, self).__init__(client)
        self.title = 'Result Queue'

    def show_queue(self, queue, namespace):
        messages = self.client.messages(queue, namespace)
        data = objective_array(messages)
        chart = plot_mean_objective_altair(data)
        return html.altair_plot(chart)
