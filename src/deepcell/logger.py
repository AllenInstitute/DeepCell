import logging


def init_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
