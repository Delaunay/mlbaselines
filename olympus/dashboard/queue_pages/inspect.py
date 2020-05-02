from olympus.dashboard.page import Page
import olympus.dashboard.elements as html


class InspectQueue(Page):
    base_path = 'queue/inspect'

    def __init__(self, client):
        self.client = client
        self.title = 'Inspect Queue'

    def routes(self):
        return [
            f'/{self.base_path}',
            f'/{self.base_path}/<string:queue>',
            f'/{self.base_path}/<string:queue>/<string:namespace>',
        ]

    def list_queues(self):
        return html.ul(html.link(q, f'/{self.base_path}/{q}') for q in self.client.queues())

    def list_namespaces(self, queue):
        return html.div(
            html.header(queue, level=4),
            html.ul(html.link(n, f'/{self.base_path}/{queue}/{n}') for n in self.client.namespaces(queue)))

    def show_queue(self, queue, namespace):
        messages = self.client.messages(queue, namespace)
        return html.div(
            html.header('/'.join([queue, namespace]), level=4),
            html.show_messages(messages))

    def main(self, queue=None, namespace=None):
        if queue is None:
            return self.list_queues()

        if namespace is None:
            return self.list_namespaces(queue)

        return self.show_queue(queue, namespace)
