from olympus.dashboard.pages.page import Page
from olympus.dashboard.render.altair_plot import AltairPlot

import pandas as pd


class DebugPage(Page):
    @staticmethod
    def route():
        return '/debug'

    def __init__(self):
        self.plot = AltairPlot(pd.DataFrame(), dom_id='debug-altair')

    def events(self, app):
        self.plot.events(app)

    def render(self, app):
        return self.plot.render(app)
