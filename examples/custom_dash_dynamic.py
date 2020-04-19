import altair as alt
from vega_datasets import data

from olympus.dashboard import Dashboard, Page, bind, set_attribute
import olympus.dashboard.elements as html


source = data.cars()
columns = list(source.columns)


class MyDynamicPage(Page):
    def routes(self):
        return '/'

    def __init__(self):
        self.title = 'MyDynamicPage'
        self.x_label = None
        self.y_label = None

    def set_attribute_callback(self, name):
        def set_attribute(value):
            # value is the index of the columns
            setattr(self, name, columns[value])

            # check if we can generate the graph
            self.make_graph()

        return set_attribute

    def make_form(self):
        """Make a simple form so the user can input the x and y axis"""
        form = html.div(
            html.div(
                html.header('X axis', level=5),
                html.select_dropdown(columns, 'x-label')),
            html.div(
                html.header('Y axis', level=5),
                html.select_dropdown(columns, 'y-label')))

        bind('x-label', 'change', self.set_attribute_callback('x_label'), property='selectedIndex')
        bind('y-label', 'change', self.set_attribute_callback('y_label'), property='selectedIndex')
        return form

    def make_graph(self):
        """Generate the graph when all the inputs are ready"""
        if self.x_label is None or self.y_label is None:
            return

        chart = alt.Chart(source).mark_circle().encode(
            alt.X(self.x_label, type='quantitative'),
            alt.Y(self.y_label, type='quantitative'),
            color='Origin:N'
        ).properties(
            width=500,
            height=500
        ).interactive()

        # send our graph back to the page
        set_attribute('graph_id', 'srcdoc', html.altair_plot(chart, with_iframe=False))

    def main(self):
        return html.div(
            self.make_form(),
            # where our graph will be populated
            html.iframe("", id='graph_id'))


if __name__ == '__main__':

    #  go to http://127.0.0.1:5000/
    dash = Dashboard(__name__)
    dash.add_page(MyDynamicPage())
    dash.run()
