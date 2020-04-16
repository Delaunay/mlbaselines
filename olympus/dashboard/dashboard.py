from argparse import ArgumentParser
import os
import re

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

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
        s = route.pattern.find('?')
        items.append(html.Li(dcc.Link(route.pattern, href=route.pattern[:s])))

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


def _make_dashboard(theme_name):
    """Make a multi-page dashboard"""
    app = dash.Dash(
        __name__,
        assets_folder=os.path.join(os.path.dirname(__file__), 'assets'),
        eager_loading=False,
        external_stylesheets=[
            f'https://stackpath.bootstrapcdn.com/bootswatch/4.4.1/{theme_name}/bootstrap.min.css'
        ]
    )
    app.config['suppress_callback_exceptions'] = True

    app.layout = html.Div([
        dcc.Location(id='url', refresh=True),
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


def dashboard(theme):
    """Global Dashboard application"""
    global _dash

    if _dash is None:
        _dash = _make_dashboard(theme)

    return _dash


themes = {
    'cerulean': 'light', 'cosmo': 'light',
    'cyborg': 'dark'   , 'darkly': 'dark',
    'flatly': 'light'  , 'litera': 'light',
    'lux': 'light'     , 'slate': 'dark',
    'solar': 'dark'    , 'superhero': 'dark'
}


def main(args=None):
    from olympus.dashboard.render import MenuRender
    from olympus.dashboard.pages import Template
    from olympus.dashboard.pages import ProjectPage, GroupPage, TrialPage, DebugPage
    from olympus.dashboard.pages import QueuePage, GanttPage, ResultPage, StatusPage, LogPage, SpaceExploration
    from olympus.dashboard.pages import FANOVAPage
    from olympus.dashboard.base import insert_kv
    from msgqueue.backends import new_monitor

    # queue_uri = 'mongo://0.0.0.0:8123'
    queue_uri = 'mongo://127.0.0.1:27017'
    # queue_uri = 'cockroach://0.0.0.0:8123'
    # queue_uri = 'zip:/home/setepenre/work/olympus/data.zip'

    parser = ArgumentParser()
    parser.add_argument('--theme', type=str, default='darkly', choices=list(themes.keys()),
                        help='CSS theme to load')
    parser.add_argument('--uri', type=str, default=queue_uri,
                        help='URI pointing to the resource to connect to\n'
                        'Examples:\n'
                        '   - mongodb instance: mongo://127.0.0.1:27017\n'
                        '   - cockroach db instance cockroach://0.0.0.0:8123\n'
                        '   - local archive: zip:/home/setepenre/work/olympus/data.zip\n')
    parser.add_argument('--database', type=str, default='olympus',
                        help='Name of the database')
    args = parser.parse_args(args)

    insert_kv('theme', args.theme)
    insert_kv('is_dark', themes.get(args.theme) == 'dark')

    dash = dashboard(args.theme)
    client = new_monitor(args.uri, args.database)

    menu = MenuRender(
        Status='/queue/status',
        Raw='/queue/raw',
        Result='/queue/result',
        Gantt='/queue/gantt',
        # Track='/project',
        Debug='/debug'
    )

    # protocol = get_protocol(args.storage_uri)
    # add_page(dash, Template(menu, ProjectPage(protocol)))
    # add_page(dash, Template(menu, GroupPage(protocol)))
    # add_page(dash, Template(menu, TrialPage(protocol)))
    add_page(dash, Template(menu, DebugPage()))
    add_page(dash, Template(menu, QueuePage(client)))
    add_page(dash, Template(menu, GanttPage(client)))
    add_page(dash, Template(menu, ResultPage(client)))
    add_page(dash, Template(menu, StatusPage(client)))
    add_page(dash, Template(menu, LogPage(client)))
    add_page(dash, Template(menu, SpaceExploration(client)))
    add_page(dash, Template(menu, FANOVAPage(client)))

    dash.run_server(debug=True)


if __name__ == '__main__':
    main()

