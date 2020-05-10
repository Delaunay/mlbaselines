import argparse
import re

from msgqueue.backends import new_client

from olympus.observers.msgtracker import METRIC_QUEUE
from olympus.hpo.parallel import RESULT_QUEUE, WORK_QUEUE


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--uri', default='mongo://127.0.0.1:27017', type=str)
    parser.add_argument('--database', default='olympus', type=str)
    parser.add_argument('--namespace', type=str)

    options = parser.parse_args(argv)

    client = new_client(options.uri, options.database)

    if options.namespace is None:
        print('Found')
        print(client.db[METRIC_QUEUE].count())
        print(client.db[WORK_QUEUE].count())
        print(client.db[RESULT_QUEUE].count())

        client.db[METRIC_QUEUE].drop()
        client.db[WORK_QUEUE].drop()
        client.db[RESULT_QUEUE].drop()

        print(client.db[METRIC_QUEUE].count())
        print(client.db[WORK_QUEUE].count())
        print(client.db[RESULT_QUEUE].count())

    else:
        query = {'namespace': {'$regex': re.compile(f"^{options.namespace}", re.IGNORECASE)}}
        print('Found')
        print(client.db[METRIC_QUEUE].count(query))
        print(client.db[WORK_QUEUE].count(query))
        print(client.db[RESULT_QUEUE].count(query))

        client.db[METRIC_QUEUE].remove(query)
        client.db[WORK_QUEUE].remove(query)
        client.db[RESULT_QUEUE].remove(query)

        print('Now there is')
        print(client.db[METRIC_QUEUE].count(query))
        print(client.db[WORK_QUEUE].count(query))
        print(client.db[RESULT_QUEUE].count(query))


if __name__ == '__main__':
    main()
