from collections import defaultdict
import json
import glob

import rpcjs.elements as html
from rpcjs.page import Page

from olympus.models import Model
from olympus.utils.storage import StateStorage


class StateBrowser(Page):
    base_path = 'state/browse'

    def __init__(self, storage: StateStorage):
        self.storage = storage
        self.title = 'State Browser'
        self.meta = None

        self.specialized_routes = {
            Model: 'model/inspect'
        }

    def routes(self):
        return [
            f'/{self.base_path}',
            f'/{self.base_path}/<string:uid>',
            f'/{self.base_path}/<string:uid>/<path:keys>',
        ]

    def subroute(self, uid, *keys, base=base_path):
        if len(keys) == 0:
            return f'/{base}/{uid}'

        return f'/{base}/{uid}/{"/".join(keys)}'

    def list_uids(self):
        self.meta = self.storage.meta.load(self.storage.folder)

        # find all the states including best and init
        all_uid = defaultdict(list)
        for i in glob.glob(f'{self.storage.folder}/*.state'):
            filename = i.split('/')[-1][:-6]
            uid = filename.split('_')[-1]
            all_uid[uid].append(filename)

        data = defaultdict(set)
        for uid, args in self.meta.items():
            task = args.pop('task', 'undefined')

            others = all_uid.get(uid, set())
            data[task] = data[task].union(others)
            data[task].add(uid)

        items = []
        for task, uids in data.items():
            items.append(''.join([task, html.ul([html.link(uid, self.subroute(uid)) for uid in uids])]))

        return html.ul(items)

    def field_display(self, uid, key, state, stype, *keys):
        base_route = self.specialized_routes.get(stype, self.base_path)

        route = self.subroute(uid, *keys, str(key), base=base_route)

        return f'{html.link(key, route)} {html.code(stype)}'

    def generic_display(self, uid, state, keys=tuple()):
        types = state.get('types', dict())

        def get_types(k, v):
            return types.get(k, type(v))

        items = []
        for key, value in state.items():
            items.append(self.field_display(uid, key, value, get_types(key, value), *keys))

        return html.ul(items)

    def show_state(self, uid):
        if self.meta is None:
            self.meta = self.storage.meta.load(self.storage.folder)

        meta_uid = uid.split('_')[-1]
        meta = self.meta.get(meta_uid, dict())
        state = self.storage.load(uid)

        display = self.generic_display(uid, state)

        return html.div(
            html.header(f'State {uid}', level=4),
            html.header('Meta', level=5),
            html.pre(json.dumps(meta, indent=2)),
            html.header('State', level=5),
            display
        )

    def show_key(self, uid, *keys):
        if self.meta is None:
            self.meta = self.storage.meta.load(self.storage.folder)

        state = self.storage.load(uid)
        for k in keys:
            t = state.get(k, dict())
            if isinstance(t, dict):
                state = t
            else:
                break

        display = self.generic_display(uid, state, keys)

        return html.div(
            html.header(f'State {uid}/{"/".join(keys)}', level=4),
            display
        )

    def main(self, uid=None, keys=None):
        if uid is None:
            return self.list_uids()

        if keys is None:
            return self.show_state(uid)

        return self.show_key(uid, *keys.split('/'))
