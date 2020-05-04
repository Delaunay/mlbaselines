import json
import io
import base64
import uuid

from flask import url_for, escape


def get_resources():
    """Fetch local resource file"""
    BOOTSTRAP    = url_for('static', filename='bootstrap.min.css')
    DARKLY       = url_for('static', filename='bootstrap.darkly.min.css')
    JQUERY       = url_for('static', filename='jquery-3.4.1.slim.min.js')
    POPPER       = url_for('static', filename='popper.min.js')
    SOCKETIO     = url_for('static', filename='socket.io.js')
    BOOTSTRAP_JS = url_for('static', filename='bootstrap.min.js')
    CUSTOM_JS    = url_for('static', filename='custom.js')

    return DARKLY, BOOTSTRAP_JS, JQUERY, POPPER, SOCKETIO, CUSTOM_JS


def ul(items):
    """Generate an unordered list

    Parameters
    ----------
    items: list
        list of DOM element to make a unordered list
    """
    items = ''.join(f'<li>{i}</li>' for i in items)
    return f'<ul>{items}</ul>'


def ol(items):
    """Generate an ordered list

        Parameters
        ----------
        items: list
            list of DOM element to make a ordered list
        """
    items = ''.join(f'<li>{i}</li>' for i in items)
    return f'<ul>{items}</ul>'


def div(*items, style=None, id=None):
    """Generate a new div

    Parameters
    ----------
    items: argument list
        DOM children of this div
    """
    children = ''.join(items)
    attr = []

    if style is not None:
        attr.append(f'style="{style}"')

    if id is not None:
        attr.append(f'id="{id}"')

    attr = ' '.join(attr)
    return f'<div {attr}>{children}</div>'


def div_row(*items, style=None):
    """Generate a new div with a row class

    Parameters
    ----------
    items: argument list
        DOM children of this div
    """
    children = ''.join(items)
    attr = []

    if style is not None:
        attr.append(f'style="{style}"')

    attr = ' '.join(attr)
    return f'<div class="row" {attr}>{children}</div>'


def div_col(*items, size=None, style=None, id=None):
    """Generate a new div with a col class

    Parameters
    ----------
    items: argument list
        DOM children of this div
    """
    children = ''.join(items)
    attr = []

    if style is not None:
        attr.append(f'style="{style}"')

    if id is not None:
        attr.append(f'id="{id}"')

    attr = ' '.join(attr)
    if size is None:
        return f'<div class="col" {attr}>{children}</div>'

    return f'<div class="col-{size}" {attr}>{children}</div>'


def header(name, level=1):
    """Generate a new header

    Parameters
    ----------
    items: str
        name of the header

    level: int
        level of the header
    """
    return f'<h{level}>{name}</h{level}>'


def link(name, ref):
    """Generate a hyperlink

    Parameters
    ----------
    name: str
        name of the link

    ref: str
        reference of the link
    """
    return f'<a href="{ref}">{name}</a>'


def span(name):
    """Generate a new span

    Parameters
    ----------
    name: str
        DOM children of this div
    """
    return f'<span>{name}</span>'


def code(name):
    """Generate a new code

    Parameters
    ----------
    name: str
        DOM children of this div
    """
    return f'<code>{escape(name)}</code>'


def chain(*args):
    """Concatenate a list of DOM elements together

    Parameters
    ----------
    items: argument list
        DOM elements
    """
    return ''.join(args)


def pre(v):
    """Generate a new pre

    Parameters
    ----------
    v: str
        DOM children of this div
    """
    return f'<pre>{escape(v)}</pre>'


def base_page(title, header, body, footer):
    """Base HTML5 page

    Parameters
    ----------
    title: str
        Title of the page

    header: str
        Header section of the page

    body: str
        Body section of the page

    footer: str
        Footer section of the page
    """
    DARKLY, BOOTSTRAP_JS, JQUERY, POPPER, SOCKETIO, CUSTOM_JS = get_resources()
    return f"""
    <!doctype html>
    <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
            <link rel="stylesheet" href="{DARKLY}">
            <title>{title}</title>
        </head>
        <body class="container-fluid">
            <header>{header}</header>
            {body}
            <footer>{footer}</footer>
            <script src="{JQUERY}"></script>
            <script src="{POPPER}"></script>
            <script src="{BOOTSTRAP_JS}"></script>
            <script src="{SOCKETIO}"></script>
            <script src="{CUSTOM_JS}"></script>
        </body>
    </html>
    """


def show_messages(messages):
    """Generate a new table displaying msqqueue messages

    Parameters
    ----------
    messages: List[Messages]
        list of messages
    """

    def make_row(m):
        try:
            data = json.dumps(m.message, indent=2)
        except TypeError as e:
            data = f'Not json serializable {e}'

        return f"""
        <tr>
            <td>{m.uid}</td>
            <td>{m.time}</td>
            <td>{m.mtype}</td>
            <td>{m.read}</td>
            <td>{m.read_time}</td>
            <td>{m.actioned}</td>
            <td>{m.actioned_time}</td>
            <td>{m.replying_to}</td>
            <td><pre>{data}</pre></td>
        </tr>
        """
    rows = ''.join([make_row(r) for r in messages])
    return f"""
    <table class="table">
        <thead>
            <th>uid</th>
            <th>time</th>
            <th>mtype</th>
            <th>read</th>
            <th>read_time</th>
            <th>actioned</th>
            <th>actioned_time</th>
            <th>replying_to</th>
            <th>message</th>
        </thhead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """


def show_agent(agents):
    """Generate a new table displaying msqqueue agents

    Parameters
    ----------
    agents: List[Agent]
        list of agents
    """
    def make_row(m):
        return f"""
        <tr>
            <td>{m.uid}</td>
            <td>{m.time}</td>
            <td>{m.agent}</td>
            <td>{m.alive}</td>
            <td>{m.namespace}</td>
            <td>{m.message}</td>
        </tr>
        """
    rows = ''.join([make_row(r) for r in agents])
    return f"""
    <table class="table">
        <thead>
            <th>uid</th>
            <th>time</th>
            <th>name</th>
            <th>alive</th>
            <th>namespace</th>
            <th>message</th>
        </thhead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """


def menu_item(name, href):
    return f'<li class="nav-item"><a href="{href}" class="nav-link">{name}</a></li>'


def menu(*items):
    list_items = ''.join(menu_item(name, link) for name, link in items)
    return f'<ul class="navbar-nav mr-auto">{list_items}</ul>'


def navbar(**kwargs):
    html_menu = menu(*kwargs.items())
    return f"""
    <div class="mb-3">
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <a href='/' class="navbar-brand">Olympus</a>
            {html_menu}
        </nav>
    </div>"""


def select_dropdown(options, id):
    """Generate a new select dropdown form

    Parameters
    ----------
    options: List[str]
        list of options the user can select

    id: str
        DOM id used to refer to this input form
    """
    html_options = ''.join(f'<option>{opt}</option>' for opt in options)
    return f"""
    <select class="form-control form-control-lg" id="{id}">
        {html_options}
    </select>
    """


def iframe(html, id=None):
    attr = []
    if id is not None:
        attr.append(f'id="{id}"')

    attr = ''.join(attr)
    return f"""
        <div style="width: 100%; height: 100%;">
            <iframe 
                {attr}
                style="position: absolute; width: 98%; height: 99%;"
                frameborder="0"
                sandbox="allow-scripts" 
                srcdoc="{escape(html)}">
            </iframe>
        </div>
        """


def altair_plot(chart, with_iframe=True):
    """Export an altair chart figure into HTML format"""
    buffer = io.StringIO()
    chart.save(buffer, 'html')
    html = buffer.getvalue()

    if not with_iframe:
        return html

    return iframe(html)


def plotly_plot(figure, full_html=False):
    """Export a plotly figure into HTML format"""
    import plotly.io

    buffer = io.StringIO()
    plotly.io.write_html(figure, buffer, auto_play=False, full_html=full_html)
    html = buffer.getvalue()

    return html


def pyplot_plot(figure, **save_args):
    """Export a matplotlib figure into HTML format"""
    out_img = io.BytesIO()
    figure.savefig(out_img, format='png', **save_args)
    figure.clf()

    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    uri = "data:image/png;base64,{}".format(encoded)
    return f"""<img src="{uri}"/>"""


def spinner():
    return """
<div class="text-center">
  <div class="spinner-border" role="status">
    <span class="sr-only">Loading...</span>
  </div>
</div>
    """


def iframe_spinner():
    return base_page('', '', spinner(), '')


def show_exception():
    import traceback
    return pre(traceback.format_exc())


# Helpers to random Plots asynchronously
def _async_plot(callback, *args, plot_id, to_html, **kwargs):
    from olympus.dashboard.dash import set_attribute
    from olympus.hpo.parallel import make_remote_call

    try:
        chart = callback(*args, **kwargs)

        if chart is None:
            return make_remote_call(
                set_attribute, plot_id, 'srcdoc', 'Not available')

        html_chart = to_html(chart)

        return make_remote_call(
            set_attribute, plot_id, 'srcdoc', html_chart)

    except:
        return make_remote_call(
            set_attribute, plot_id, 'srcdoc', show_exception())


def _make_async_plot(to_html):
    def _plot_async_base(fun, *args, id=None, **kwargs):
        from olympus.dashboard.dash import async_call

        if id is None:
            id = uuid.uuid4().hex[0:16]

        async_call(_async_plot, fun, *args, plot_id=id, to_html=to_html, **kwargs)

        # Returns an empty iframe
        return iframe(iframe_spinner(), id=id)
    return _plot_async_base


def plotly_plot_full(plot):
    return plotly_plot(plot, full_html=True)


async_altair_plot = _make_async_plot(altair_plot)
async_plotly_plot = _make_async_plot(plotly_plot_full)
async_pyplot_plot = _make_async_plot(pyplot_plot)
