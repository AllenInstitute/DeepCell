import abc
from typing import Dict

import numpy as np


class Callback(abc.ABC):
    """Base callback"""
    def on_epoch_end(self, epoch: int, metrics: Dict[str, np.ndarray]):
        """Call to update the callback state or perform some action

        Args:
            epoch:
                Index of epoch
            metrics:
                Dict with key metric name and value array of metric value at
                each epoch.
        """
        raise NotImplementedError
