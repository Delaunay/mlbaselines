from track.structure import Project, TrialGroup, Trial

from dash.development.base_component import Component
import dash_html_components as html
import dash_core_components as dcc

from typing import TypeVar

from olympus.dashboard.base import DOMComponent
from .utils import dict_to_html

ListItem = TypeVar('ListItem')


def pydict_key(k):
    return html.Span(k, className='pydict_key')


def pydict_value(v):
    return html.Span(v, className='pydict_value')


class ProjectRender(DOMComponent):
    def __init__(self, project: Project):
        self.project = project

    def __repr__(self):
        return self.project.name

    @property
    def name(self) -> ListItem:
        return html.H5(f'Project: {self.project.name}')

    @property
    def description(self) -> ListItem:
        return html.Li([pydict_key('Description'), pydict_value(self.project.description)])

    @property
    def metadata(self) -> ListItem:
        meta = self.project.metadata
        if not meta:
            return None

        return html.Li([pydict_key('Metadata'), dict_to_html(meta)])

    @property
    def groups(self) -> ListItem:
        groups_children = []

        for group in self.project.groups:
            groups_children.append(
                html.Li(dcc.Link(group.name, href=f'/group/{group.uid}')))

        groups = html.Ul(groups_children)
        return html.Li([pydict_key('Groups'), groups])

    def render(self, app) -> Component:
        return html.Div(
            className='track-project',
            children=[html.Ul(children=[
                self.name,
                self.description,
                self.metadata,
                self.groups
            ])]
        )


class TrialGroupRender(DOMComponent):
    def __init__(self, group: TrialGroup):
        self.group = group

    def __repr__(self):
        return self.group.name

    @property
    def name(self) -> ListItem:
        return html.Li([pydict_key('Name'), pydict_value(self.group.name)])

    @property
    def description(self) -> ListItem:
        return html.Li([pydict_key('Description'), pydict_value(self.group.description)])

    @property
    def metadata(self) -> ListItem:
        return html.Li([pydict_key('Metadata'), dict_to_html(self.group.metadata)])

    @property
    def trials(self) -> ListItem:
        trials_children = []

        for trial in self.group.trials:
            trials_children.append(
                html.Li(dcc.Link(trial, href=f'/trial/{trial}')))

        trials = html.Ul(trials_children)
        return html.Li([pydict_key('Trials'), trials])

    @property
    def project(self) -> ListItem:
        return html.Li(dcc.Link('Project', href=f'/project/{self.group.project_id}'))

    def render(self, app) -> Component:
        return html.Div(
            className='track-group',
            children=[html.Ul(children=[
                self.project,
                self.name,
                self.description,
                self.metadata,
                self.trials
            ])]
        )


class TrialRender(DOMComponent):
    def __init__(self, trial: Trial):
        self.trial = trial

    def __repr__(self):
        return self.trial.name

    @property
    def name(self) -> ListItem:
        return html.Li([pydict_key('Name'), pydict_value(self.trial.name)])

    @property
    def description(self) -> ListItem:
        return html.Li([pydict_key('Description'), pydict_value(self.trial.description)])

    @property
    def metadata(self) -> ListItem:
        return html.Li([pydict_key('Metadata'), dict_to_html(self.trial.metadata)])

    @property
    def metrics(self) -> ListItem:
        return html.Li([pydict_key('Metrics'), dict_to_html(self.trial.metrics)])

    @property
    def parameters(self) -> ListItem:
        return html.Li([pydict_key('Parameters'), dict_to_html(self.trial.parameters)])

    @property
    def chronos(self) -> ListItem:
        return html.Li([pydict_key('Chrono'), dict_to_html(self.trial.chronos)])

    @property
    def errors(self) -> ListItem:
        return html.Li([pydict_key('Errors'), pydict_value(self.trial.errors)])

    @property
    def status(self) -> ListItem:
        return html.Li([pydict_key('Status'), pydict_value(self.trial.status.name)])

    @property
    def uid(self) -> ListItem:
        return html.Li([pydict_key('UID'), pydict_value(self.trial.uid)])

    @property
    def tags(self) -> ListItem:
        return html.Li([pydict_key('Tags'), dict_to_html(self.trial.tags)])

    @property
    def project(self) -> ListItem:
        return html.Li(dcc.Link('Project', href=f'/project/{self.trial.project_id}'))

    @property
    def group(self) -> ListItem:
        return html.Li(dcc.Link('Group', href=f'/group/{self.trial.group_id}'))

    def render(self, app) -> Component:
        return html.Div(
            className='track-trial',
            children=[html.Ul(children=[
                self.project,
                self.group,
                self.uid,
                self.name,
                self.description,
                self.metadata,
                self.parameters,
                self.metrics,
                self.chronos,
                self.errors,
            ])]
        )
