import inspect
import logging
from time import time

logging.basicConfig(level=logging.INFO)
default_logger = logging.getLogger(__name__)


def measure_time(logger=default_logger, log_params=False):
    def timeit(method):
        def timed(*args, **kw):
            ts = time()
            result = method(*args, **kw)
            te = time()

            msg = f"{method.__name__} executed in {te-ts:.2f} s"

            signature = inspect.signature(method).parameters.keys()
            params = [f"{k}: {v}" for k, v in zip(signature, args)] + [
                f"{k}: {v}" for k, v in kw.items()
            ]
            if log_params:
                msg += f"\nwith params: {' === '.join(params)}"

            logger.info(msg)
            return result

        return timed

    return timeit
