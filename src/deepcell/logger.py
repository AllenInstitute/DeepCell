import logging
import sys


def init_logger(name):
    logger = logging.getLogger(name)
    streamHandler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - '
                                  '%(message)s')
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.setLevel(logging.INFO)
    return logger
