from olympus.dashboard.queue_pages.inspect import InspectQueue
import olympus.dashboard.elements as html


class LogsQueue(InspectQueue):
    base_path = 'queue/logs'

    def __init__(self, client):
        self.client = client
        self.title = 'Logs Queue'

    def routes(self):
        return [
            f'/{self.base_path}',
            f'/{self.base_path}/<string:queue>',
            f'/{self.base_path}/<string:queue>/<string:namespace>',
            f'/{self.base_path}/<string:queue>/<string:namespace>/<string:agent_id>',
        ]

    def main(self, queue=None, namespace=None, agent_id=None):
        if queue is None:
            return self.list_queues()

        if namespace is None:
            return self.list_namespaces(queue)

        if agent_id is None:
            return self.list_agents(queue, namespace)

        return self.show_queue(queue, namespace, agent_id)

    def list_agents(self, queue, namespace):
        return html.div(
            html.header('Agents', level=4),
            html.ul(
                html.link(q, f'/{self.base_path}/{queue}/{namespace}/{q.uid}') for q in self.client.agents()))

    def show_queue(self, queue, namespace, agent_id):
        log = self.client.log(agent_id)
        return html.pre(log)

