import logging
from typing import Optional


def init_logger(name, log_path: Optional[str] = None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if log_path is not None:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'))
        logger.addHandler(fh)
    return logger
