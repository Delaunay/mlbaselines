from dash.development.base_component import Component
import dash_html_components as html
import dash_core_components as dcc

from olympus.dashboard.base import DOMComponent


class MenuRender(DOMComponent):
    def __init__(self, **kwargs):
        self.menu_list = self.make_menu_list(kwargs)

    def menu_item(self, name, href):
        return html.Li(className='nav-item', children=[
            dcc.Link(name, className='nav-link', href=href)
        ])

    def make_menu_list(self, items):
        return html.Ul(className='navbar-nav mr-auto', children=[
            self.menu_item(name, link) for name, link in items.items()
        ])

    def render(self, app) -> Component:
        return html.Div(className='mb-3', children=html.Nav(
            className='navbar navbar-expand-lg navbar-dark bg-dark',
            children=[
                dcc.Link('Olympus', className='navbar-brand', href='/'),
                self.menu_list
            ]
        ))
