import logging
from enum import Enum
from typing import Optional, Any
import datetime


class LogLevel(Enum):
    DEBUG = 1
    ERROR = 2
    WARNING = 3
    INFO = 4


class Logger:
    def __init__(self, name: str, log_path: Optional[str] = None,
                 level: LogLevel = LogLevel.INFO):
        self._name = name
        self._log_path = log_path
        self._level = level

    def _write(self, level: LogLevel, message: Any):
        timestamp = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        to_log = f'{timestamp} | {self._name} | {level.name} | {message}'
        if level.value >= self._level.value:
            print(to_log)
            if self._log_path is not None:
                with open(self._log_path, mode='a') as f:
                    f.write(f'\n{to_log}')

    def info(self, message: Any):
        self._write(level=LogLevel.INFO, message=message)
