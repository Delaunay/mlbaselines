from olympus.dashboard.render import TrialRender
from olympus.dashboard.pages.page import Page


class TrialPage(Page):
    @staticmethod
    def route():
        return '/trial/(?P<trial_id>[a-zA-Z0-9_]*)'

    def __init__(self, protocol):
        self.protocol = protocol
        self.trial = None

    def render(self, app, trial_id):
        trials = self.protocol.fetch_trials({'uid': trial_id})

        if not trials:
            return f'(trial_id: {trial_id}) does not exist'

        return TrialRender(trials[0]).render(app)
