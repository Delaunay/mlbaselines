class Trainer:
    def __init__(self):
        pass

    def fit(self, epochs, dataloader, *args, **kwargs):
        """Train the model for `epochs`"""
        raise NotImplementedError()

    def step(self, batch, *args, **kwargs):
        """Compute loss for a single batch, do a bit of setup before calling `batch`"""
        raise NotImplementedError()

    def batch(self, batch, *args, **kwargs):
        """Compute loss for a single batch"""
        raise NotImplementedError()

    def save(self, path, override=True):
        """Save training state to a file to enable resuming"""
        raise NotImplementedError()

    def resume(self, path, find_latest=True):
        """Resume training from a state dictionary"""
        raise NotImplementedError()
