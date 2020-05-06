from flask import Flask
from flask_socketio import SocketIO

from olympus.utils import debug, info, error
from olympus.hpo.parallel import exec_remote_call

import traceback
from threading import Thread
from multiprocessing import Process, Pool

_socketio = None
_socketio_ready = False
_host = None
_port = None
_pool = None


def host():
    return _host


def port():
    return _port


def socketio():
    return _socketio


def _decorated_mp(host, port, fun, *args, **kwargs):
    import socketio

    r = None
    exception = None
    try:
        r = fun(*args, **kwargs)
    except:
        exception = traceback.format_exc()

    if 'module' in r and 'function' in r:
        # make a client to the flask server
        socket = socketio.Client()
        socket.connect(f'http://{host}:{port}')

        if r is not None:
            socket.emit('remote_process_result', r)

        if exception is not None:
            socket.emit('remote_process_error', exception)

    else:
        error(f'(result: {r}) is is not a remote call')


def async_thread_call(fun, *args, **kwargs):
    """Runs a function async and execute the resulting remote function call"""
    t = Thread(
        target=_decorated_mp,
        args=(host(), port(), fun) + args,
        kwargs=kwargs)

    t.start()
    return t


def async_process_call(fun, *args, **kwargs):
    p = Process(
        target=_decorated_mp,
        args=(host(), port(), fun) + args,
        kwargs=kwargs)

    p.start()
    return p


def async_call(fun, *args, **kwargs):
    if _pool is not None:
        return _pool.apply_async(
            _decorated_mp,
            args=(host(), port(), fun) + args,
            kwds=kwargs)
    else:
        error('Pool was not created, restart your server')


def register_event(event, handler, namespace='/'):
    """Register a socketio event to a python handler

    Parameters
    ----------
    event: str
        Name of the event

    handler: call
        Function to call when the event is fired
    """
    return socketio().on(event, namespace)(handler)


_pending_bind = []
_pending_attr = []


def set_attribute(id, attribute, value):
    """Set the attribute of an element on the webpage

    Parameters
    ----------
    id: str
        id of the DOM element

    attribute: str
        name of the attribute to set

    value: json
        new value of the attribute
    """
    if not _socketio_ready:
        _pending_attr.append((id, attribute, value))
        return

    debug(f'set_attribute {attribute} of {id}')

    socketio().emit('set_attribute', dict(
        id=id,
        attribute=attribute,
        value=value
    ))


def redirect(url):
    """Set the attribute of an element on the webpage

    Parameters
    ----------
    url: str
        url to redirect the client to

    """
    debug(f'redirect ot {url}')
    socketio().emit('redirect', dict(url=url))


def set_html(id, value):
    """Set the attribute of an element on the webpage

    Parameters
    ----------
    id: str
        id of the DOM element

    value: json
        new value of the attribute
    """
    debug(f'set_html of {id}')

    socketio().emit('set_html', dict(
        id=id,
        html=value
    ))


def get_element_size(id, callback):
    """Get the size of an element inside the webpage

    Parameters
    ----------
    id: str
        id of the DOM element

    callback: Call
        Function to call with the size information `{width: w, height: h}`
    """
    debug(f'get_element_size of {id}')
    socketio().emit('get_size', dict(
        id=id
    ))
    register_event(f'get_size_{id}', callback)


def bind(id, event, handler, attribute=None, property=None):
    """Bind an element event to a handler and return a property of an attribute of the element

    Parameters
    ----------
    id: str
        id of the DOM element

    event: str
        Name of the event we are listening too.
        The full list of supported events can be found `here <https://www.w3schools.com/jsref/dom_obj_event.asp>`_.

    handler: call
        function to callback when the event is fired

    attribute: str
        Attribute of the element to return

    property: str
        Property of the element to return
    """
    if not _socketio_ready:
        _pending_bind.append((id, event, handler, attribute, property))
        return

    debug(f'binding `{id}` with `{event}` to `{handler}`')
    # ask javascript to listen to events for a particular kind of event on our element
    socketio().emit('bind', {'id': id, 'event': event, 'attribute': attribute, 'property': property})
    # when the event happen js will send us back the innerHTML of that element
    register_event(f'bind_{event}_{id}', handler)
    return


def handshake_event():
    """Called when socketIO connects to the server"""
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
    """Called when socketIO disconnects from the server"""
    global _socketio_ready
    _socketio_ready = False
    info('SocketIO disconnected')


def handle_error(msg):
    print(msg)


class Dashboard:
    """Dashboard entry point"""
    def __init__(self, name=__name__, secret='__secret__'):
        import os
        self.app = Flask(name, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
        self.app.debug = True
        self.app.config['SECRET_KEY'] = secret
        self.socket = SocketIO(self.app)
        self.routes = []

        global _socketio
        _socketio = self.socket

        self.on_event('handshake', handshake_event)
        self.on_event('disconnect', disconnect_event)
        self.on_event('remote_process_result', exec_remote_call)
        self.on_event('remote_process_error', handle_error)

    @staticmethod
    def init_worker():
        # Make the worker ignore KeyboardInterrupts
        # since the master process is going to do the cleanup
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def __enter__(self):
        global _pool
        _pool = Pool(4, self.init_worker)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if _pool is not None:
            _pool.close()
            _pool.terminate()

    def run(self, host='127.0.0.1', port=5000):
        """Run the flask App"""
        global _host, _port

        _port = port
        _host = host
        return self.socket.run(self.app, host, port)

    def add_page(self, page, route=None, header=None, **kwargs):
        """Add a new page to the dashboard

        Parameters
        ----------
        page: Page
            page object

        route: str
            Route specification to reach the page
            Will default to the page route if left undefined

        header: str
            HTML header to insert onto the page
        """
        route = route or page.routes()

        if not isinstance(route, list):
            route = [route]

        for r in route:
            self.routes.append((r, type(page).__name__))
            self.app.add_url_rule(r, type(page).__name__, page, **kwargs)

        if header is not None:
            page.header = header

    def on_event(self, event, handler, namespace='/'):
        """Register an handler for a givent event"""
        self.socket.on(event, namespace)(handler)

