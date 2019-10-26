
from mlbaselines.distributed.mongo import MongoDB


if __name__ == '__main__':
    db = MongoDB('localhost', 8123, location='/tmp/mongodb')
    db.start()

    db.stop()
