import argparse
import datetime
from pprint import pprint
import re
from collections import defaultdict
import threading

from msgqueue.backends import new_client

from olympus.hpo.optimizer import HyperParameterOptimizer
from olympus.observers.msgtracker import METRIC_QUEUE
from olympus.hpo.parallel import RESULT_QUEUE, WORK_QUEUE, HPO_ITEM, WORK_ITEM, RESULT_ITEM
from olympus.hpo.worker import TrialWorker
from olympus.utils.log import set_verbose_level
from olympus.studies.searchspace.main import get_hpo


def get_hpo_status(client, namespace, queue, item):
    count = client.db[queue].count({'namespace': namespace, 'mtype': item})
    pending = client.monitor().unread_count(queue, namespace, mtype=item)
    read = client.monitor().read_count(queue, namespace, mtype=item)
    actioned = client.monitor().actioned_count(queue, namespace, mtype=item)
    return dict(
        count=count,
        pending=pending,
        running=read - actioned,
        completed=actioned)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--uri', default='mongo://127.0.0.1:27017', type=str)
    parser.add_argument('--database', default='olympus', type=str)
    parser.add_argument('--namespace', default=None, type=str)
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--show-errors', action='store_true')

    args = parser.parse_args(argv)

    set_verbose_level(3)

    client = new_client(args.uri, args.database)

    query = {'namespace': {'$regex': re.compile(f"^{args.namespace}", re.IGNORECASE)}}
    stats = client.db[WORK_QUEUE].aggregate([
        {'$match': query},
        {'$project': {
            'namespace': 1,
       }},
        {'$group': {
            '_id': '$namespace',
        }},
    ])
    stats = sorted(doc['_id'] for doc in stats)

    if not stats:
        print(f'No namespace found for {args.namespace}')
        return 0

    if len(stats) > 1:
        print('\n'.join(stats))
        print('All these namespaces were found.')
        namespaces = stats
    else:
        namespaces = [args.namespace]

    for namespace in namespaces:
        print()
        print(namespace)
        if args.show_errors:
            show_errors(client, namespace, HPO_ITEM)
            show_errors(client, namespace, WORK_ITEM)
        repair_hpo_duplicates(client, namespace, test_only=args.test_only)
        repair_trials_duplicates(client, namespace, test_only=args.test_only)
        repair_hpo_lost_results(client, args.uri, args.database, namespace, test_only=args.test_only)
        failover_broken(client, namespace, test_only=args.test_only)


def repair_hpo_duplicates(client, namespace, test_only=False):
    stats = get_hpo_status(client, namespace, WORK_QUEUE, HPO_ITEM)

    # Reset HPOs
    if stats['count'] - stats['completed'] > 1:
        print('ERROR: {} duplicates found'.format(stats['count'] - stats['completed'] - 1))


        hpos = client.db[WORK_QUEUE].find(
            {'namespace': namespace, 'mtype': HPO_ITEM, 'actioned': False})

        uids = set()
        sizes = dict()
        for hpo_message in hpos:
            # deserialize state to get trials
            hpo = HyperParameterOptimizer(None, None, seed=0)
            hpo.load_state_dict(hpo_message['message']['hpo_state'])
            hpo_uids = set(hpo.trials.keys())
            # HPO has new uids, but some of past uids are missing, thus it diverged.
            if (hpo_uids - uids) and (uids - hpo_uids):
                print('ERROR: hpo duplicate {hpo["_id"]} diverged')
            sizes[hpo_message["_id"]] = len(hpo_uids)
            uids = uids | hpo_uids

        hpo_to_keep = next(iter(sorted(sizes.items(), key=lambda i: i[1], reverse=True)))

        print(f'     would keep {hpo_to_keep[0]}')

        if test_only:
            return

        client.db[WORK_QUEUE].update_many(
            {'namespace': namespace, 'mtype': HPO_ITEM},
            {'$set': {'read': True, 'actioned': True}})

        client.db[WORK_QUEUE].find_one_and_update(
            {'_id': hpo_to_keep[0]},
            {'$set': {'read': False, 'actioned': False}})

    elif stats['count'] - stats['completed'] == 1:
        print('OK: Only 1 actionable hpo found')
    else:
        print('OK: HPO completed')


def show_errors(client, namespace, mtype):

    cursor = client.db[WORK_QUEUE].find(
        {'namespace': namespace, 'mtype': mtype, 'actioned': False})

    print(datetime.datetime.utcnow())
    for i, trial in enumerate(cursor):
        if trial.get('error') is None:
            continue

        trial_uid = trial['message']['kwargs']['uid']
        print(i, trial_uid, trial['_id'], trial['read'], trial['actioned'])
        error = trial.pop('error')
        pprint(trial)
        print('Retries:', trial['retry'])
        print(error)


def repair_trials_duplicates(client, namespace, test_only=False):
    cursor = client.db[WORK_QUEUE].find(
        {'namespace': namespace, 'mtype': WORK_ITEM, 'actioned': False},
        {'message.kwargs.uid': 1})

    trials = defaultdict(list)

    for trial in cursor:
        trial_uid = trial['message']['kwargs']['uid']
        trials[trial_uid].append(trial['_id'])

    error = False
    for trial_uid, message_uids in trials.items():
        if len(message_uids) > 1:
            print(f'ERROR: {len(message_uids)} duplicates for trial {trial_uid}')
            error = True
            if not test_only:
                for message_id in message_uids[1:]:
                    client.db[WORK_QUEUE].delete_one({'_id': message_id})
    if not error:
        print('OK: No trial duplicate')


def failover_broken(client, namespace, test_only=False):
    if test_only:
        return
    # Verify retry and error in uncompleted trials
    client.db[WORK_QUEUE].update_many(
        {'namespace': namespace, 'mtype': WORK_ITEM, 'actioned': False},
        {'$set': {'retry': 0}})

    client.db[WORK_QUEUE].update_many(
        {'namespace': namespace, 'mtype': HPO_ITEM, 'actioned': False},
        {'$set': {'retry': 0}})


def repair_hpo_lost_results(client, uri, database, namespace, test_only=False):

    stats = get_hpo_status(client, namespace, WORK_QUEUE, HPO_ITEM)

    if namespace == 'sst2_hpo-bayesopt-s-10':
        client.db[RESULT_QUEUE].update_many(
            {'namespace': namespace, 'mtype': RESULT_ITEM},
            {'$set': {'read': False, 'actioned': False}})


    trials = client.db[RESULT_QUEUE].find(
        {'namespace': namespace, 'mtype': RESULT_ITEM, 'actioned': True},
        {'message.uid': 1})

    # Do this instead of count to avoid duplicates
    n_completed_and_observed = len(set(doc['message'][0]['uid'] for doc in trials))

    # Reset HPOs
    if stats['count'] == stats['completed']:
        hpo = get_hpo(client, namespace)[0]
        n_observed = sum(1 for trial in hpo.trials.values() if trial.objective is not None)

        if n_observed < n_completed_and_observed:
            print('ERROR: HPO lost {} observations and we cannot restore!!!'.format(n_completed_and_observed - n_observed))
        else:
            print('OK: No HPO observation lost')
    else:
        hpo = get_hpo(client, namespace, partial=True)[0]
        n_observed = sum(1 for trial in hpo.trials.values() if trial.objective is not None)

        if n_observed < n_completed_and_observed:
            print('ERROR: HPO lost {} observations'.format(n_completed_and_observed - n_observed))
            if test_only:
                return
            # Update trials
            client.db[RESULT_QUEUE].update_many(
                {'namespace': namespace, 'mtype': RESULT_ITEM},
                {'$set': {'read': False, 'actioned': False}})

            # Reset trial in work queue (if lost)

            # Run HPO to get it completed or in a repaired state where it can sample.
            worker = TrialWorker(
                uri, database, 1, namespace,
                hpo_allowed=True, work_allowed=False)
            worker.timeout = 1

            # Make sur it only runs the HPO once
            def stop():
                worker.running = False
            threading.Timer(2, stop).start()

            worker.run()
        else:
            print('OK: No HPO observation lost')


if __name__ == '__main__':
    main()
