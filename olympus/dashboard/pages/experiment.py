from olympus.dashboard.page import Page
from olympus.hpo.parallel import WORK_QUEUE, RESULT_QUEUE
from olympus.observers.msgtracker import METRIC_QUEUE
import olympus.dashboard.elements as html

from olympus.dashboard.queue_pages import MetricQueue, StatusQueue, FANVOAQueue, SpaceQueue, GanttQueue, InspectQueue


class ExperimentOverview(Page):
    base_path = 'experiment'

    def __init__(self, client):
        self.client = client
        self.title = 'Experiment'
        self.sections = {
            'status': self.status,
            'metrics': self.metrics,
            'fanova': self.fanova,
            'exploration': self.exploration,
            'work_distribution': self.work_distribution,
            'Raw': self.inspect
        }

    def routes(self):
        return [
            f'/{self.base_path}',
            f'/{self.base_path}/<string:experiment>',
            f'/{self.base_path}/<string:experiment>/<string:section>',
        ]

    def sidebar(self, name):
        return html.sidebar(
            **{k.capitalize():  f'/{self.base_path}/{name}/{k}' for k in self.sections.keys()})

    def list_namespaces(self, queue=WORK_QUEUE):
        return html.div(
            html.header('Experiments', level=4),
            html.ul(html.link(n, f'/{self.base_path}/{n}') for n in self.client.namespaces(queue)))

    def main(self, experiment=None, section=None):
        if experiment is None:
            return self.list_namespaces()

        # call the section render function
        section_html = self.sections.get(
            section, self.no_section)(experiment, section)

        return html.div_col(
            html.header(experiment),
            html.div_row(
                self.sidebar(experiment),
                html.div_col(section_html, classes='col py-2')), classes="container-fluid")

    def no_section(self, experiment, section):
        if section is None:
            return ''

        return f'No section named {section}'

    def status(self, experiment, section):
        return StatusQueue(self.client).main(WORK_QUEUE, experiment)

    def metrics(self, experiment, section):
        return MetricQueue(self.client).main(METRIC_QUEUE, experiment)

    def fanova(self, experiment, section):
        return FANVOAQueue(self.client).main(RESULT_QUEUE, experiment)

    def exploration(self, experiment, section):
        return SpaceQueue(self.client).main(WORK_QUEUE, experiment)

    def work_distribution(self, experiment, section):
        return GanttQueue(self.client).main(WORK_QUEUE, experiment)

    def inspect(self, experiment, section):
        return InspectQueue(self.client).main(WORK_QUEUE, experiment)

