from olympus.utils.fp16 import OptimizerAdapter


class OptimizerBuilder:
    def __call__(self, model_parameters, weight_decay, half=False, loss_scale=1,
                 dynamic_loss_scale=False, scale_window=1000, scale_factor=2,
                 min_loss_scale=None, max_loss_scale=2.**24, **kwargs):
        return OptimizerAdapter(
            self.build(model_parameters, weight_decay, **kwargs),
            half=half,
            loss_scale=loss_scale,
            dynamic_loss_scale=dynamic_loss_scale,
            scale_window=scale_window,
            scale_factor=scale_factor,
            min_loss_scale=min_loss_scale,
            max_loss_scale=max_loss_scale
        )

    def build(self, model_parameters, weight_decay, **kwargs):
        raise NotImplementedError

    def get_space(self):
        raise NotImplementedError

    def get_params(self, params):
        optimizer_params = dict()

        for key in self.get_space().keys():
            optimizer_params[key] = params[key]

        return optimizer_params
