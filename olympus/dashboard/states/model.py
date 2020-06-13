import traceback

import rpcjs.elements as html
from rpcjs.page import Page
import rpcjs.binded as js
from rpcjs.binding import set_attribute

import torch.nn as nn

from olympus.models import Model
from olympus.utils.storage import StateStorage
from olympus.dashboard.plots.layer import vizualize_param


def list_layers(model):
    layers = []
    for k, module in model.named_modules():
        if type(module) != nn.Sequential and k != '':
            if k[0] == '_':
                k = k[1:]

            layers.append((k, module))

    return layers[1:]


class InspectModel(Page):
    base_path = 'model/inspect'

    def __init__(self, storage: StateStorage):
        self.storage = storage
        self.title = 'Inspect Model'
        self.uid = None
        self.keys = None
        self.module = None
        self.param = None
        self.data = None
        self.inputs = []

    def routes(self):
        return [
            f'/{self.base_path}/<string:uid>/<path:keys>',
            f'/{self.base_path}/<string:uid>/<path:keys>/m/<string:module>',
            f'/{self.base_path}/<string:uid>/<path:keys>/m/<string:module>/p/<string:param>',
            f'/{self.base_path}/<string:uid>/<path:keys>/m/<string:module>/p/<string:param>/d/<string:uid2>'
        ]

    def load_model(self, uid, keys):
        state = self.storage.load(uid)
        for k in keys:
            state = state.get(k, dict())

        return Model.from_state(state)

    def module_subroute(self, layer):
        return f'/{self.base_path}/{self.uid}/{self.keys}/m/{layer}'

    def module_selection(self, model: nn.Module, uid, keys):
        layers = []
        for k, v in list_layers(model):
            layers.append(' '.join([
                html.link(k, self.module_subroute(k)), html.code(type(v))
            ]))

        return html.div(
            html.header(f'Modules of {html.code(".".join(keys))} in {html.code(uid)}', level=4),
            html.ol(layers))

    def param_subroute(self, param, uid):
        base = f'/{self.base_path}/{self.uid}/{self.keys}/m/{self.module}/p/{param}'
        if uid is None:
            return base

        return f'{base}/d/{uid}'

    def parameter_link(self, name, param, uid2=None):
        if len(param.shape) == 0:
            size = param.item()
        else:
            size = param.shape

        return ' '.join([
            html.link(name, self.param_subroute(name, uid2)),
            html.code(size)
        ])

    def parameter_selection(self, model, module_name):
        layers = dict(list_layers(model))
        layer = layers[module_name]
        params = layer.named_parameters()

        return html.div_row(
            html.div_col(
                html.header(f'Module {module_name}', level=4),
                html.pre(str(layer)),
                html.header('Parameters', level=4),
                html.ul([self.parameter_link(k, v) for k, v in params]),
                style="height: 100vh;"),
            html.div_col())

    def update_plot(self):
        end = self.data
        for i in self.inputs:
            v = i.get()
            if not v:
                v = 0
            else:
                v = int(v)

            end = end[v]

        fig = vizualize_param(end)

        set_attribute('weight_plot', 'srcdoc', html.pyplot_plot(fig, with_iframe=False))

    def vizualize_param(self, param):
        param = param.squeeze()
        shape = param.shape

        # Simple Heatmap/bar will done
        if len(shape) <= 2:
            fig = vizualize_param(param)
            return html.pyplot_plot(fig)

        # Need to select dimensions to analyze
        self.data = param
        html_forms = []
        self.inputs = []

        for s in shape[:-2]:
            v, html_input = js.number_input(min=0, max=s - 1, callback=self.update_plot)
            self.inputs.append(v)
            html_forms.append(html_input)

        return html.div_row(
            html.div_col(html.chain(*html_forms)),
            html.iframe("", id='weight_plot'), style="height: 100vh;")

    def show_param(self, model, module_name, param_name, model2, uid2):
        if model2 is not None:
            layers = dict(list_layers(model2))
            layer = layers[module_name]
            params = dict(layer.named_parameters())
            param2 = params[param_name]

        layers = dict(list_layers(model))
        layer = layers[module_name]
        params = dict(layer.named_parameters())
        param = params[param_name]

        if model2 is not None:
            param = param - param2

        title = param_name
        if model2 is not None:
            title = f'{param_name} - {uid2}'

        return html.div_row(
            html.div_col(
                html.header(f'Module {module_name}', level=4),
                html.pre(str(layer)),
                html.header('Parameters', level=4),
                html.ul([self.parameter_link(k, v, uid2) for k, v in params.items()])
            ),
            html.div_col(
                html.header(title, level=4),
                self.vizualize_param(param))
        )

    def main(self, uid, keys, module=None, param=None, uid2=None):
        self.uid = uid
        self.keys = keys
        self.module = module
        self.param = param
        model2 = None

        keys = keys.split('/')
        try:
            model = self.load_model(uid, keys)

            if uid2 is not None:
                model2 = self.load_model(uid2, keys)
        except:
            return html.div(
                f'Could not load model in {uid}.state {".".join(keys)}',
                html.pre(traceback.format_exc())
            )

        if module is None:
            return self.module_selection(model, uid, keys)

        if param is None:
            return self.parameter_selection(model, module)

        return self.show_param(model, module, param, model2, uid2)
