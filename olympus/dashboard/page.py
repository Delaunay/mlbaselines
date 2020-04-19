import olympus.dashboard.elements as html


class Page:
    footer = ''
    header = ''

    @staticmethod
    def routes():
        raise NotImplementedError()

    def __init__(self):
        self.title = type(self).__name__
        self.header = ''
        self.footer = ''

    def __call__(self, *args, **kwargs):
        return self.base(self.main(*args, **kwargs))

    def base(self, body):
        return html.base_page(self.title, self.header, body, self.footer)

    def main(self, *args, **kwargs):
        raise NotImplementedError

