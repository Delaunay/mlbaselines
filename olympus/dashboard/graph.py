import base64
import io

import dash_html_components as html

from olympus.dashboard.base import DOMComponent


class Altair(DOMComponent):
    """Save an Altair chart to an HTML iframe"""
    def __init__(self, chart):
        chart_html = io.StringIO()
        chart.save(chart_html, 'html')
        self.html = chart_html.getvalue()

    def render(self, app=None):
        return html.Div(html.Iframe(
            srcDoc=self.html,
            height='100%',
            width='100%',
            sandbox='allow-scripts',
            style={
                'border-width': '0px',
                'position': 'absolute'
            }))


class PyPlot(DOMComponent):
    """Save a matplotlib figure to an HTML image"""
    def __init__(self, figure):
        self.figure = figure

    def fig_to_uri(self, in_fig, **save_args) -> str:
        out_img = io.BytesIO()
        in_fig.savefig(out_img, format='png', **save_args)
        # clear the current figure.
        in_fig.clf()

        out_img.seek(0)  # rewind file
        encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
        return "data:image/png;base64,{}".format(encoded)

    def render(self, app=None):
        return html.Img(src=self.fig_to_uri(self.figure))
