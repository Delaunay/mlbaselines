from olympus.hpo.parallel import WORK_ITEM, HPO_ITEM, WORKER_LEFT, WORKER_JOIN, RESULT_ITEM, SHUTDOWN


def extract_message(m):
    return m.message


def filter_result(m):
    return m.mtype == RESULT_ITEM


def extract_objective(m):
    params, result = m
    params['objective'] = result
    return params


def objective_array(messages):
    """Returns [{uid, epoch=epoch, ..., objective=value}]"""
    return map(
        extract_objective, map(
            extract_message, filter(
                filter_result, messages)))


def extract_work_messages(messages):
    """Return work messages and estimate the number of workers"""
    worker_count = 0
    worker_left = 0
    worker_shutdown = 0

    work_items = []
    for m in messages:
        if not m.read_time or not m.actioned_time:
            continue

        if m.mtype == WORKER_JOIN:
            worker_count += 1

        elif m.mtype == WORKER_LEFT:
            worker_left += 1

        elif m.mtype == HPO_ITEM or m.mtype == WORK_ITEM:
            work_items.append(m)

        elif m.mtype == RESULT_ITEM:
            pass

        elif m.mtype == SHUTDOWN:
            worker_shutdown += 1

        else:
            print(m.mtype, 'ignored')

    work_items.sort(key=lambda m: m.read_time)
    worker_count = max(worker_count, worker_left, worker_shutdown)
    return work_items, worker_count


def extract_configuration(messages, unique=False):
    """Extract a list of configuration"""
    import copy

    if not unique:
        results = []
    else:
        results = {}

    columns = set()

    # Keep the most recent trial
    for m in messages:
        if m.mtype == WORK_ITEM:
            uid = m.message['kwargs']['uid']

            params = copy.deepcopy(m.message['kwargs'])
            params.pop('uid')
            columns.update(params.keys())

            if not unique:
                results.append(params)
            else:
                old = results.get(uid)
                if old is None:
                    results[uid] = params
                elif old['epoch'] < params['epoch']:
                    results[uid] = params

    columns.discard('epoch')

    if not unique:
        values = results
    else:
        values = results.values()

    values = list(sorted(values, key=lambda p: p['epoch']))
    return values, list(columns)


def extract_last_results(messages):
    results = {}
    columns = set()

    # Keep the most recent trial
    for m in messages:
        if m.mtype == RESULT_ITEM:
            params, objective = m.message

            params = dict(params)
            params['objective'] = objective

            columns.update(params.keys())

            uid = params.get('uid')

            old_params = results.get(uid)
            if old_params is None or old_params['epoch'] < params['epoch']:
                results[uid] = params

    return list(results.values()), list(columns)
