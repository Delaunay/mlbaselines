from typing import List
import json

from dash.development.base_component import Component
import dash_html_components as html
import dash_core_components as dcc

from olympus.dashboard.base import DOMComponent

from msgqueue.backends.queue import Message, Agent


def to_json(data):
    try:
        return json.dumps(data, indent=2)
    except:
        return str(data)


class MessageRender(DOMComponent):
    def __init__(self, message: Message):
        self.msg = message

    def __repr__(self):
        return repr(self.msg)

    def render(self, app) -> Component:
        return html.Tr(children=[
            html.Td(str(self.msg.uid)),
            html.Td(str(self.msg.time)),
            html.Td(self.msg.mtype),
            html.Td(str(self.msg.read)),
            html.Td(str(self.msg.read_time)),
            html.Td(str(self.msg.actioned)),
            html.Td(str(self.msg.actioned_time)),
            html.Td(str(self.msg.replying_to)),
            html.Td(html.Pre(to_json(self.msg.message))),
        ])


class MessagesRender(DOMComponent):
    def __init__(self, messages: List[Message]):
        self.messages = messages

    def __repr__(self):
        return 'list(messages)'

    def render(self, app) -> Component:
        return html.Table(className='table', children=[
            html.Thead(children=[
                html.Th('uid'),
                html.Th('time'),
                html.Th('mtype'),
                html.Th('read'),
                html.Th('read_time'),
                html.Th('actioned'),
                html.Th('actioned_time'),
                html.Th('replying_to'),
                html.Th('message'),
            ]),
            html.Tbody(children=[
                MessageRender(m).render(app) for m in self.messages
            ])
        ])


class AgentRender(DOMComponent):
    def __init__(self, agent: Agent):
        self.agent = agent

    def __repr__(self):
        return repr(self.agent)

    def render(self, app, namespace) -> Component:
        return html.Tr(children=[
            html.Td(str(self.agent.uid)),
            html.Td(str(self.agent.time)),
            html.Td(dcc.Link(self.agent.agent, href=f'/queue/logs/{namespace}/{self.agent.uid}')),
            html.Td(str(self.agent.heartbeat)),
            html.Td(str(self.agent.alive)),
            html.Td(str(self.agent.message))
        ])


class AgentsRender(DOMComponent):
    def __init__(self, agents: List[Agent], namespace):
        self.agents = agents
        self.namespace = namespace

    def __repr__(self):
        return 'list(agents)'

    def render(self, app) -> Component:
        return html.Table(className='table', children=[
            html.Thead(children=[
                html.Th('uid'),
                html.Th('time'),
                html.Th('name'),
                html.Th('heartbeat'),
                html.Th('alive'),
                html.Th('message')
            ]),
            html.Tbody(children=[
                AgentRender(m).render(app, self.namespace) for m in self.agents
            ])
        ])
