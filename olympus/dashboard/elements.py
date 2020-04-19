import json
import io
import base64
from flask import url_for, escape


def get_resources():
    BOOTSTRAP    = url_for('static', filename='bootstrap.min.css')
    DARKLY       = url_for('static', filename='bootstrap.darkly.min.css')
    JQUERY       = url_for('static', filename='jquery-3.4.1.slim.min.js')
    POPPER       = url_for('static', filename='popper.min.js')
    SOCKETIO     = url_for('static', filename='socket.io.js')
    BOOTSTRAP_JS = url_for('static', filename='bootstrap.min.js')
    CUSTOM_JS    = url_for('static', filename='custom.js')

    return DARKLY, BOOTSTRAP_JS, JQUERY, POPPER, SOCKETIO, CUSTOM_JS


def ul(items):
    items = ''.join(f'<li>{i}</li>' for i in items)
    return f'<ul>{items}</ul>'


def ol(items):
    items = ''.join(f'<li>{i}</li>' for i in items)
    return f'<ul>{items}</ul>'


def div(*items, style=None):
    children = ''.join(items)

    if style is None:
        return f'<div>{children}</div>'

    return f'<div style="{style}">{children}</div>'


def div_row(*items):
    children = ''.join(items)
    return f'<div class="row">{children}</div>'


def div_col(*items, size=None):
    children = ''.join(items)
    if size is None:
        return f'<div class="col">{children}</div>'

    return f'<div class="col-{size}">{children}</div>'


def header(name, level=1):
    return f'<h{level}>{name}</h{level}>'


def link(name, ref):
    return f'<a href="{ref}">{name}</a>'


def span(name):
    return f'<span>{name}</span>'


def code(name):
    return f'<code>{escape(name)}</code>'


def chain(*args):
    return ''.join(args)


def pre(v):
    return f'<pre>{escape(v)}</pre>'


def base_page(title, header, body, footer):
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
    def make_row(m):
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
            <td><pre>{json.dumps(m.message, indent=2)}</pre></td>
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
    html_options = ''.join(f'<option>{opt}</option>' for opt in options)
    return f"""
    <select class="form-control form-control-lg" id="{id}">
        {html_options}
    </select>
    """


def altair_plot(chart):
    """Export an altair chart figure into HTML format"""
    buffer = io.StringIO()
    chart.save(buffer, 'html')
    html = buffer.getvalue()
    return f"""
        <iframe 
            style="border-width: 0px; position: absolute;" 
            width="100%" 
            height="100%"
            sandbox="allow-scripts" 
            srcdoc="{escape(html)}">
        </iframe>
        """


def plotly_plot(figure):
    """Export a plotly figure into HTML format"""
    import plotly.io

    buffer = io.StringIO()
    plotly.io.write_html(figure, buffer, auto_play=False, full_html=False)
    html = buffer.getvalue()

    return html

    f"""
            <iframe 
                style="border-width: 0px;" 
                width="100%" 
                height="100%"
                sandbox="allow-scripts" 
                srcdoc="{escape(html)}">
            </iframe>
            """


def pyplot_plot(figure, **save_args):
    """Export a matplotlib figure into HTML format"""
    out_img = io.BytesIO()
    figure.savefig(out_img, format='png', **save_args)
    figure.clf()

    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    uri = "data:image/png;base64,{}".format(encoded)
    return f"""<img src="{uri}"/>"""
