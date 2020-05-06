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
            f'/{self.base_path}/<string:namespace>/<string:agent_id>',
            f'/{self.base_path}/<int:ltype>/<string:agent_id>',
            f'/{self.base_path}/<string:queue>/<string:namespace>',
            f'/{self.base_path}/<string:queue>/<string:namespace>/<string:agent_id>',
        ]

    def main(self, queue=None, namespace=None, agent_id=None, ltype=0):
        if queue is None and agent_id is None:
            return self.list_queues()

        if namespace is None and agent_id is None:
            return self.list_namespaces(queue)

        if agent_id is None:
            return self.list_agents(queue, namespace)

        return self.show_queue(namespace, agent_id, ltype)

    def list_agents(self, queue, namespace):
        return html.div(
            html.header('Agents', level=4),
            html.ul(
                html.link(q, f'/{self.base_path}/{queue}/{namespace}/{q.uid}') for q in self.client.agents(namespace)))

    def show_queue(self, namespace, agent_id, ltype):
        log = self.client.log(agent_id, ltype=ltype)
        return html.div(
            html.header(f'Logs for agent {agent_id[:8]}'),
            html.pre(log))

