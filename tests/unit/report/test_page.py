SKIP_TEST=False
try:
    from olympus.dashboard.pages.group import GroupPage
    from olympus.dashboard.pages.trial import TrialPage
    from olympus.dashboard.pages.project import ProjectPage

    from track.persistence import get_protocol
except ImportError:
    SKIP_TEST = True


import pytest
import os


data_dir = os.path.dirname(__file__)
TEST_FILE = f'file://{os.path.dirname(__file__)}/simple.json'


@pytest.mark.skipif(SKIP_TEST, reason='Dependencies not installed')
def test_project_page(track_uri=TEST_FILE):
    protocol = get_protocol(track_uri)
    ProjectPage(protocol).render(
        None,
        project_id='minimalist_hpo'
    )


@pytest.mark.skipif(SKIP_TEST, reason='Dependencies not installed')
def test_group_page(track_uri=TEST_FILE):
    protocol = get_protocol(track_uri)
    GroupPage(protocol).render(
        None,
        group_id='759d3c1d107967b958e4432e4b754991600926860f21e735cccb9bcf7aca42c1'
    )


@pytest.mark.skipif(SKIP_TEST, reason='Dependencies not installed')
def test_trial_page(track_uri=TEST_FILE):
    protocol = get_protocol(track_uri)
    TrialPage(protocol).render(
        None,
        trial_id='ac8a2fb76f81ada5948bb7601f68da25_0'
    )

