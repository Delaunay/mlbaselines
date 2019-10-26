from mlbaselines.distributed.cockroachdb import start_message_queue, CKMQClient
from mlbaselines.utils import debug


def test_cockroach():
    db = start_message_queue(
        'orion',
        '/tmp/mq',
        addrs='192.168.0.10:8123',
        join=None,
        clean_on_exit=True
    )
    db.start()
    db.new_queue('orion')

    debug('Initialized')
    client = CKMQClient('cockroach://192.168.0.10:8123')
    debug('Client Ready')

    for i in range(0, 20):
        client.push('orion', {'id': i})
        debug('push')

        assert client.unread_count('orion') == 1
        m = client.pop('orion')
        debug('pop')

        if m is not None:
            assert m.message['id'] == i
            a = client.unactioned_count('orion')
            assert client.unactioned_count('orion') == (i + 1)

        assert client.unread_count('orion') == 0

    # for i in range(0, 21):
    m = client.mark_actioned('orion', m)
    print(m)

    # client.dump('orion')
    db.stop()
    debug('stop')


if __name__ == '__main__':
    test_cockroach()
