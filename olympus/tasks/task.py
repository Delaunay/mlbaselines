import torch


class Task:
    def __init__(self, device=None):
        self._device = device if device else torch.device('cpu')

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        for name in dir(self):
            attr = getattr(self, name)
            if hasattr(attr, 'to'):
                setattr(self, name, attr.to(device=device))

        self._device = device

    def fit(self, step, input, context):
        """Execute a single batch

        Parameters
        ----------

        step: int
            current batch_id

        input: Tensor. Tuple[Tensor]
            Pytorch tensor or tuples of tensors

        context: dict
            Optional Context
        """
        raise NotImplementedError()

    @property
    def metrics(self):
        pass

    def report(self, pprint=True, print_fun=print):
        m = self.metrics
        if m:
            return self.metrics.report(pprint, print_fun)

    def finish(self):
        m = self.metrics
        if m:
            return self.metrics.finish(self)
