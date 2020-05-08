import rpcjs.elements as html
from rpcjs.page import Page
from rpcjs.binding import bind, redirect

from olympus.hpo.parallel import WORK_QUEUE, RESULT_QUEUE
from olympus.observers.msgtracker import METRIC_QUEUE
from olympus.dashboard.queue_pages import MetricQueue, StatusQueue, GanttQueue, InspectQueue
from olympus.utils import debug


class StudyOverview(Page):
    base_path = 'study'

    def __init__(self, client):
        self.client = client
        self.title = 'Study'
        self.prefix = None
        self.delimiter = '-'
        self.sections = {
            'status': self.status,
            'metrics': self.metrics,
            'work_distribution': self.work_distribution,
            'Raw': self.inspect
        }

    def routes(self):
        return [
            f'/{self.base_path}',
            f'/{self.base_path}/<string:study>',
            f'/{self.base_path}/<string:study>/<string:section>',
        ]

    def sidebar(self, name):
        return html.sidebar(
            **{k.capitalize():  f'/{self.base_path}/{name}/{k}' for k in self.sections.keys()})

    def select_study(self):
        def redirect_on_enter(data):
            if not data:
                return
            self.prefix = data
            redirect(f'/{self.base_path}/{data}')

        bind('study_prefix', 'change', redirect_on_enter, property='value')

        return html.div(
            html.header('Study prefix', level=5),
            'A study is a group of experiment, experiment belonging to the same study should'
            'start with the same name for example <code>mystudy_exp1</code>, <code>my_study_exp2</code>',
            html.text_input(id='study_prefix', placeholder='Study prefix')
        )

    def main(self, study=None, section=None):
        if study is None:
            return self.select_study()

        self.title = f'Study {study.capitalize()}'
        section_html = self.sections.get(
            section, self.no_section)(study, section)

        return html.div_col(
            html.header(f'Study: {study.capitalize()}'),
            html.div_row(
                self.sidebar(study),
                html.div_col(section_html, classes='col py-2')), classes="container-fluid")

    def no_section(self, experiment, section):
        if section is None:
            return ''

        return f'No section named {section}'

    def status(self, study, section):
        return StatusQueue(self.client).show_overview(WORK_QUEUE, study, delimiter=self.delimiter)

    def metrics(self, study, section):
        return MetricQueue(self.client.aggregate_monitor()).show_queue(METRIC_QUEUE, study, delimiter=self.delimiter)

    def work_distribution(self, study, section):
        return GanttQueue(self.client).show_queue(WORK_QUEUE, study, delimiter=self.delimiter, color='g1')

    def inspect(self, study, section):
        return InspectQueue(self.client.aggregate_monitor()).show_queue(WORK_QUEUE, study, delimiter=self.delimiter)

