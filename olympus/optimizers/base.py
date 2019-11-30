

class OptimizerBuilder:
    def __call__(self, model_parameters, weight_decay, **kwargs):
        return self.build(model_parameters, weight_decay, **kwargs)

    def build(self, model_parameters, weight_decay, **kwargs):
        raise NotImplementedError

    def get_space(self):
        raise NotImplementedError

    def get_params(self, params):
        optimizer_params = dict()

        for key in self.get_space().keys():
            optimizer_params[key] = params[key]

        return optimizer_params

    @staticmethod
    def defaults():
        return {}
