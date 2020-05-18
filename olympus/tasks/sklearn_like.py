from olympus.tasks.task import Task
from olympus.observers import ElapsedRealTime, SampleCount
from olympus.utils import HyperParameters, drop_empty_key


class SklearnTask(Task):
    def __init__(self, model):
        super(SklearnTask, self).__init__()
        self.model = model

        # Measure the time spent training
        self.metrics.append(ElapsedRealTime().every(batch=1))
        self.metrics.append(SampleCount().every(batch=1))

    def get_space(self):
        """Return hyper parameter space of the task"""
        # in that simple case only the model has HP
        # but in DL you would have the optimizer, lr-scheduler, etc...
        return drop_empty_key({
            'model': self.model.get_space(),
        })

    def init(self, model, uid=None):
        self.model.init(**model)

        # Get a unique identifier for this configuration
        if uid is None:
            uid = compute_identity(model, size=16)

        # broadcast a signal that the model is ready
        # so we can setup logging, data, etc...
        self.metrics.new_trial(model, uid)

    def fit(self, x, y, epoch=None, context=None):
        # broadcast to observers/metrics that we are starting the training
        self.metrics.start_train()
        self.metrics.new_epoch(0)
        self.metrics.new_batch(0, input=x)

        self.model.fit(x, y)

        self.metrics.end_batch(1, input=x)
        self.metrics.end_epoch(1)
        # broadcast to observers/metrics that we are ending the training
        self.metrics.end_train()

    def accuracy(self, x, y):
        # How to measure accuracy given our model
        pred = self.model.predict(x)
        accuracy = (pred == y).mean()

        # We expect accuracy and loss
        return accuracy, 0

    def auc(self, x, y):
        # How to measure accuracy given our model
        preds = self.model.predict(x)
	fpr, tpr, _  = roc_curve(y, preds)
	auc_result = auc(fpr,tpr)

        pcc = np.corrcoef(preds, targets)[0, 1]

	return auc_result, pcc

    # If you support resuming implement those methods
    def load_state_dict(self, state, strict=True):
        pass

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        pass


