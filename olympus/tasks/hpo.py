from olympus.tasks.task import Task
from olympus.utils import show_dict
from olympus.hpo import HPOptimizer, fidelity


class HPO(Task):
    """
    Attributes
    ----------
    hpo: HPOptimizer
        Hyper parameter optimizer to find the best hyper parameter

    task_maker: Call[Task]
        A function that generate a task to do hyper parameter optimization on
    """
    def __init__(self, hpo, task_maker):
        super(HPO, self).__init__()
        self.hpo = hpo
        self.task_maker = task_maker

    def fit(self, objective):
        """Train the model a few times and return a best trial/set of parameters"""

        self.metrics.start_train()
        while not self.hpo.is_done():
            configurations = self.hpo.suggest()

            for config in configurations:
                show_dict(config)

                # uid = config.pop('uid')
                epoch = config.pop('epoch')

                new_task = self.task_maker()
                new_task.init(**config)
                new_task.fit(epoch)

                metrics = new_task.metrics.value()
                result = metrics[objective]

                # config['uid'] = uid
                self.hpo.observe(config, result)

        self.metrics.end_train()
        return self.hpo.result()

    def is_done(self):
        return self.hpo.is_done()

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = super(HPO, self).state_dict(destination, prefix, keep_vars)
        state['hpo'] = self.hpo.state_dict()
        return state

    def load_state_dict(self, state, strict=True):
        super(HPO, self).load_state_dict(state, strict)
        self.hpo.load_state_dict(state['hpo'])
        return self




if __name__ == '__main__':
    pass
