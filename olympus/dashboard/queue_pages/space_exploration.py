from olympus.dashboard.queue_pages.inspect import InspectQueue
from olympus.dashboard.queue_pages.utilities import extract_configuration
from olympus.dashboard.plots.hyperparameter_exploration import scatter_matrix_altair
from olympus.dashboard.elements import altair_plot


class SpaceQueue(InspectQueue):
    base_path = 'queue/space'

    def __init__(self, client):
        super(SpaceQueue, self).__init__(client)
        self.title = 'Result Queue'

    def show_queue(self, queue, namespace):
        messages = self.client.messages(queue, namespace)
        configs, columns = extract_configuration(messages)
        return altair_plot(scatter_matrix_altair(configs, columns))
