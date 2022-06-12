import logging
from typing import Optional


def init_logger(name, log_path: Optional[str] = None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    is_filehandler_already_added = \
        len([x for x in logger.handlers if type(x) is logging.FileHandler]) > 0
    if log_path is not None and not is_filehandler_already_added:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'))
        logger.addHandler(fh)
    return logger
