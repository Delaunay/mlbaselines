import dash_core_components as dcc
import dash_html_components as html

from olympus.dashboard.render import ProjectRender
from olympus.dashboard.pages.page import Page


class ProjectPage(Page):
    @staticmethod
    def route():
        return '/project/?(?P<project_id>[a-zA-Z0-9]*)'

    def events(self, app):
        return

    def __init__(self, protocol):
        self.protocol = protocol
        self.projects = None

    def render(self, app, project_id):
        if project_id == '':
            project_id = None

        if self.projects is None:
            self.projects = self.protocol.fetch_projects({})

        return html.Div(
            className='project-view row',
            children=[
                self.project_list(),
                self.project_details(app, project_id)
            ]
        )

    def project_list(self):
        projects_children = []
        for p in self.projects:
            projects_children.append(
                html.Li(dcc.Link(p.name, href=f'/project/{p.name}')))

        projects = html.Ul(projects_children)

        return html.Div(
            className='projects-list col-md-auto',
            children=[
                html.H4('Projects:'),
                projects
            ])

    def project_details(self, app, project_id):
        if project_id:
            p = self.get_project(project_id)

            if p is None:
                return f'Project {project_id} does not exist'

            return ProjectRender(p).render(app)

        return html.Div(id='project-details')

    def get_project(self, name):
        for p in self.projects:
            if p.name == name:
                return p
        return None