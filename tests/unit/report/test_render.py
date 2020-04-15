SKIP_TEST=False
try:
    from olympus.dashboard.render.altair_plot import AltairPlot
    from olympus.dashboard.render.simple_plot import SimplePlot
    from olympus.dashboard.render.track import TrialGroupRender, TrialRender, ProjectRender
    from olympus.dashboard.render.utils import list_to_html, dict_to_html, to_html
except ImportError:
    SKIP_TEST = True

import pytest
import pandas as pd


@pytest.mark.skipif(SKIP_TEST, reason='Dependencies not installed')
def test_altair():
    AltairPlot(pd.DataFrame()).render(None)


@pytest.mark.skipif(SKIP_TEST, reason='Dependencies not installed')
def test_simple_plot():
    SimplePlot(pd.DataFrame()).render(None)


@pytest.mark.skipif(SKIP_TEST, reason='Dependencies not installed')
def test_trial_group():
    from track.structure import TrialGroup
    TrialGroupRender(TrialGroup()).render(None)


@pytest.mark.skipif(SKIP_TEST, reason='Dependencies not installed')
def test_trial():
    from track.structure import Trial
    TrialRender(Trial()).render(None)


@pytest.mark.skipif(SKIP_TEST, reason='Dependencies not installed')
def test_project():
    from track.structure import Project
    ProjectRender(Project()).render(None)


@pytest.mark.skipif(SKIP_TEST, reason='Dependencies not installed')
def test_list_to_html():
    list_to_html([1, 2, 3, 4])


@pytest.mark.skipif(SKIP_TEST, reason='Dependencies not installed')
def test_dict_to_html():
    dict_to_html({'a': 1, 'b': "b"})


@pytest.mark.skipif(SKIP_TEST, reason='Dependencies not installed')
def test_to_html():
    to_html({'a': {'a': 1, 'b': "b"}, 'b': [1, 2, 3, 4]})
