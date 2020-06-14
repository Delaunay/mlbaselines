import logging
import sys

from olympus.utils.options import option, set_option

verbose_log_mapping = [
    logging.WARN,   # 0
    logging.WARN,   # 1
    logging.INFO,   # 2
    logging.DEBUG   # 3
]


def set_verbose_level(level):
    """Set verbose level
        - 0 disables all progress output (warning logging enabled)
        - 1 enables progress output and warning logging
        - 2 adds info logging
        - 3 adds debug logging
    """
    if level <= 0:
        # mute progress printing
        set_option('progress.frequency_epoch', 0)
        set_option('progress.frequency_batch', 0)

    if level >= len(verbose_log_mapping):
        level = -1

    set_log_level(verbose_log_mapping[level])


def set_log_level(level=logging.INFO):
    oly_log.setLevel(level)


def get_log_record_constructor():
    old_factory = logging.getLogRecordFactory()

    def log_record(name, level, path, lno, msg, args, exc_info, func=None, sinfo=None, **kwargs):
        start = path.rfind('/olympus/')
        if start > -1:
            path = path[start + 1:]
        return old_factory(name, level, path, lno, msg, args, exc_info, func, sinfo, **kwargs)

    return log_record


if globals().get('oly_log') is None:
    logging.basicConfig(
        level=option('logging.level', logging.WARN, type=int),
        format='%(asctime)s [%(levelname)8s] %(name)s [%(process)d] %(pathname)s:%(lineno)d %(message)s',
        stream=sys.stdout
    )

    oly_log = logging.getLogger('OLYMPUS')
    logging.setLogRecordFactory(get_log_record_constructor())

    warning = oly_log.warning
    info = oly_log.info
    debug = oly_log.debug
    error = oly_log.error
    critical = oly_log.critical
    exception = oly_log.exception
