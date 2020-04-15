import dash_html_components as html

from olympus.dashboard.pages.page import Page


class Template(Page):
    def __init__(self, header, page, footer=None):
        self.header = header
        self.page = page
        self.footer = footer

    def route(self):
        return self.page.route()

    def events(self, app):
        return self.page.events(app)

    def render(self, app, *args, **kwargs):
        return html.Div(className='container-fluid', children=[
            self.header.render(app),
            self.page.render(app, *args, **kwargs)
        ])
