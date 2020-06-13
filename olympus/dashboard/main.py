import logging
from olympus.utils.log import set_log_level
set_log_level(logging.DEBUG)

from argparse import ArgumentParser

from olympus.dashboard.queue_pages import InspectQueue, SpaceQueue, ResultQueue, MetricQueue
from olympus.dashboard.queue_pages import GanttQueue, FANVOAQueue, LogsQueue, StatusQueue
from olympus.dashboard.pages.experiment import ExperimentOverview
from olympus.dashboard.pages.study import StudyOverview

from olympus.dashboard.states import StateBrowser, InspectModel

from olympus.utils import option
from olympus.utils.storage import StateStorage

import rpcjs.elements as html
from rpcjs.dashboard import Dashboard
from rpcjs.page import Page


class MainPage(Page):
    """Display available routes the user can take"""
    @staticmethod
    def routes():
        return '/'

    def __init__(self, dash):
        super(MainPage, self).__init__()
        self.dash = dash
        self.title = 'main'

    def main(self):
        routes = [
            html.link(html.chain(html.span(name), ':', html.code(spec)), spec) for spec, name in self.dash.routes]
        return html.div(
            html.header('Routes', level=4),
            html.ul(routes))


def dashboard():
    """Dashboard entry point for Olympus user"""
    from msgqueue.backends import new_monitor

    parser = ArgumentParser()
    parser.add_argument('--uri', type=str, default='mongo://127.0.0.1:27017',
                        help='URI pointing to the resource to connect to\n'
                             'Examples:\n'
                             '   - mongodb instance: mongo://127.0.0.1:27017\n'
                             '   - cockroach db instance cockroach://0.0.0.0:8123\n'
                             '   - local archive: zip:/home/setepenre/work/olympus/data.zip\n')
    parser.add_argument('--database', type=str, default='olympus',
                        help='Name of the database')
    parser.add_argument('--storage', type=str, default=option('state.storage', '/home/setepenre/zshare/tmp'),
                        help='FS path to storage')
    args = parser.parse_args()

    dash = Dashboard()
    client = new_monitor(args.uri, args.database)

    state_storage = StateStorage(folder=args.storage)

    navbar = html.navbar(
        Experiments='/experiment',
        Studies='/study',
        States='/state/browse',
        Debug='/',
    )

    dash.add_page(MainPage(dash), header=navbar)
    dash.add_page(StatusQueue(client), header=navbar)
    dash.add_page(InspectQueue(client), header=navbar)
    dash.add_page(ResultQueue(client), header=navbar)
    dash.add_page(GanttQueue(client), header=navbar)
    dash.add_page(SpaceQueue(client), header=navbar)
    dash.add_page(FANVOAQueue(client), header=navbar)
    dash.add_page(LogsQueue(client), header=navbar)
    dash.add_page(MetricQueue(client), header=navbar)
    dash.add_page(ExperimentOverview(client), header=navbar)
    dash.add_page(StudyOverview(client), header=navbar)
    dash.add_page(StateBrowser(state_storage), header=navbar)
    dash.add_page(InspectModel(state_storage), header=navbar)

    return dash


def main():
    with dashboard() as dash:
        dash.run()


if __name__ == '__main__':
    main()
