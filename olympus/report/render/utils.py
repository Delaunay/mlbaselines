import dash_html_components as html

from track.serialization import to_json


def prettify_name(name):
    return name.replace('_', ' ').capitalize()


def to_html(obj, depth=0):
    if isinstance(obj, dict):
        return dict_to_html(obj, depth + 1)

    elif isinstance(obj, list):
        return list_to_html(obj, depth + 1)

    if depth == 0:
        return to_html(to_json(obj), depth + 1)

    return obj


def dict_to_html(obj, depth=0):
    children = []
    for k, v in obj.items():
        children.append(
            html.Li([
                html.Span(k, className='pydict_key'),
                html.Span(to_html(v, depth + 1), className='pydict_value')
            ], className='pydict_item')
        )

    return html.Ul(children, className='pydict')


def list_to_html(obj, depth=0):
    return html.Ul(
        [html.Li(to_html(i, depth + 1), className='pylist_value') for i in obj],
        className='pylist'
    )

