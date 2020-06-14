from rpcjs.binded import realtime_altair_plot

from olympus.dashboard.queue_pages.inspect import InspectQueue
from olympus.dashboard.queue_pages.utilities import objective_array
from olympus.dashboard.plots.training_curve import plot_mean_objective_altair
from olympus.dashboard.queue_pages.utilities import fetch_new_messages


class ResultQueue(InspectQueue):
    base_path = 'queue/result'

    def __init__(self, client):
        super(ResultQueue, self).__init__(client)
        self.title = 'Result Queue'

    def show_queue(self, queue, namespace):
        chart = plot_mean_objective_altair([], objective='val_loss')
        chart = chart.interactive()

        update_fetcher = fetch_new_messages(
            self.client.state_dict(),
            queue,
            namespace,
            objective_array)

        return realtime_altair_plot(chart, update_fetcher)

