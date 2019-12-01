from argparse import ArgumentParser
import os
import re

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from track.persistence import get_protocol

from olympus.utils import info, set_verbose_level


set_verbose_level(1000)

_dash = None
_routing = {}
_regex_routing = {}
_route_cache = {}


class DuplicateRoute(Exception):
    pass


def default_index():
    items = []
    for route, _ in _routing.items():
        items.append(html.Li(dcc.Link(route, href=route)))

    for route, _ in _regex_routing.items():
        items.append(html.Li(dcc.Link(route.pattern, href=route.pattern)))

    return html.Div([
        html.H4('Routes:'),
        html.Ul(items)
    ])


def router(app, pathname):
    # Have to do route caching because of Dash sending too many similar requests
    # for no reasons
    global _route_cache

    if pathname is None:
        return ''

    if pathname in _routing:
        info(f'Route {pathname}')
        return _routing[pathname](app)

    if pathname in _route_cache:
        return _route_cache[pathname]()

    # Apply REGEX routes
    for regex, handler in _regex_routing.items():
        result = re.match(regex, pathname)

        if result is not None:
            args = result.groupdict()
            info(f'Route {pathname} matched with {regex.pattern} and (args: {args})')

            lazy = lambda: handler(app, **args)
            _route_cache[pathname] = lazy
            return lazy()

    if pathname == '/':
        return default_index()

    return '404 Page not found'


def _make_dashboard():
    """Make a multi-page dashboard"""
    app = dash.Dash(
        __name__,
        assets_folder=os.path.join(os.path.dirname(__file__), 'assets'),
        external_stylesheets=[
            'https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css'
        ]
    )
    app.config['suppress_callback_exceptions'] = True

    app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-content')
    ])

    def app_router(pathname):
        return router(app, pathname)

    app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])(app_router)
    return app


def add_route(route_path, route_handler):
    """Insert a route to the dashboard"""
    global _routing

    if route_path in _routing:
        raise DuplicateRoute(route_path)

    _routing[route_path] = route_handler


def add_regex_route(route_regex, route_handler):
    """Insert a regex route to the dashboard

    Examples
    --------

    >>> def project_handler(app, project_id):
    >>>     return project_id, group_id
    >>>
    >>> add_regex_route(
    >>>     '/project/?(?P<project_id>[a-zA-Z0-9]*)',
    >>>     project_handler)

    """
    global _regex_routing

    compiled = re.compile(route_regex)
    if compiled in _regex_routing:
        raise DuplicateRoute(route_regex)

    _regex_routing[compiled] = route_handler


def register_events(app, obj):
    obj.events(app)


def add_page(app, page):
    add_regex_route(page.route(), page.render)
    page.events(app)


def dashboard():
    """Global Dashboard application"""
    global _dash

    if _dash is None:
        _dash = _make_dashboard()

    return _dash


def main(args=None):
    from olympus.report.pages import ProjectPage, GroupPage, TrialPage, DebugPage

    dash = dashboard()

    track_uri = 'file:/home/setepenre/work/olympus-run/track_test.json'

    parser = ArgumentParser(args)
    parser.add_argument('--storage-uri', type=str, default=track_uri)

    args = parser.parse_args(args)
    protocol = get_protocol(args.storage_uri)

    # Manual Routing
    # --------------
    # add_regex_route(
    #    '/project/?(?P<project_id>[a-zA-Z0-9]*)',
    #    project_list.render
    # )

    add_page(dash, ProjectPage(protocol))
    add_page(dash, GroupPage(protocol))
    add_page(dash, TrialPage(protocol))
    add_page(dash, DebugPage())

    dash.run_server(debug=True)


if __name__ == '__main__':
    main()

