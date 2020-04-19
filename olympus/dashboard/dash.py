from flask import Flask
from flask_socketio import SocketIO

from olympus.utils import debug, info

_socketio = None
_socketio_ready = False


def socketio():
    return _socketio


def register_event(event, handler, namespace='/'):
    return socketio().on(event, namespace)(handler)


_pending_bind = []
_pending_attr = []


def set_attribute(id, attribute, value):
    if not _socketio_ready:
        _pending_bind.append((id, attribute, value))

    debug(f'set_attribute {attribute} of {id} to {value}')

    socketio().emit('set_attribute', dict(
        id=id,
        attribute=attribute,
        value=value
    ))


def bind(id, event, handler, attribute=None, property=None):
    if not _socketio_ready:
        _pending_bind.append((id, event, handler, attribute, property))

    debug(f'binding `{id}` with `{event}` to `{handler}`')
    # ask javascript to listen to events for a particular kind of event on our element
    socketio().emit('bind', {'id': id, 'event': event, 'attribute': attribute, 'property': property})
    # when the event happen js will send us back the innerHTML of that element
    register_event(f'bind_{event}_{id}', handler)
    return


def handshake_event():
    global _socketio_ready, _pending_attr, _pending_bind

    _socketio_ready = True
    info('SocketIO connected')

    if _pending_attr:
        for arg in _pending_attr:
            set_attribute(*arg)

        _pending_attr = []

    if _pending_bind:
        for arg in _pending_bind:
            bind(*arg)

        _pending_bind = []


def disconnect_event():
    global _socketio_ready
    _socketio_ready = False
    info('SocketIO disconnected')


class Dashboard:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.debug = True
        self.app.config['SECRET_KEY'] = 'secret!'
        self.socket = SocketIO(self.app)
        self.routes = []

        global _socketio
        _socketio = self.socket

        self.on_event('handshake', handshake_event)
        self.on_event('disconnect', disconnect_event)

    def run(self):
        return self.socket.run(self.app)

    def add_page(self, page, route=None, header=None, **kwargs):
        route = route or page.routes()

        if not isinstance(route, list):
            route = [route]

        for r in route:
            self.routes.append((r, type(page).__name__))
            self.app.add_url_rule(r, type(page).__name__, page, **kwargs)

        if header is not None:
            page.header = header

    def on_event(self, event, handler, namespace='/'):
        self.socket.on(event, namespace)(handler)

