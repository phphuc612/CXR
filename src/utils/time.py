import logging
from time import time

logging.basicConfig(level=logging.INFO)
default_logger = logging.getLogger(__name__)


def measure_time(logger=default_logger):
    def timeit(method):
        def timed(*args, **kw):
            ts = time()
            result = method(*args, **kw)
            te = time()
            logger.info(f"{method.__name__} executed in {te-ts:.2f} s")
            return result

        return timed

    return timeit
