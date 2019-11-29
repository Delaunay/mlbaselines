from olympus.report.render.altair_plot import AltairPlot
from olympus.report.render.simple_plot import SimplePlot
from olympus.report.render.track import TrialGroupRender, TrialRender, ProjectRender
from olympus.report.render.utils import list_to_html, dict_to_html, to_html

import pandas as pd


def test_altair():
    AltairPlot(pd.DataFrame()).render(None)


def test_simple_plot():
    SimplePlot(pd.DataFrame()).render(None)


def test_trial_group():
    from track.structure import TrialGroup
    TrialGroupRender(TrialGroup()).render(None)


def test_trial():
    from track.structure import Trial
    TrialRender(Trial()).render(None)


def test_project():
    from track.structure import Project
    ProjectRender(Project()).render(None)


def test_list_to_html():
    list_to_html([1, 2, 3, 4])


def test_dict_to_html():
    dict_to_html({'a': 1, 'b': "b"})


def test_to_html():
    to_html({'a': {'a': 1, 'b': "b"}, 'b': [1, 2, 3, 4]})
