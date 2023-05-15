import numpy as np
import torch
import torchvision.transforms.functional as TF
import random


class RandomRotate90:
    """Rotates by a random multiple of 90 degrees"""
    def __call__(self, x: torch.tensor):
        n_rotatations = random.choice(range(-3, 4))
        angle = 90 * n_rotatations
        return TF.rotate(x, angle)


class ReverseVideo:
    """Reverses video"""
    def __init__(self, p):
        self._p = p

    def __call__(self, x: torch.tensor):
        if random.random() < self._p:
            return np.flip(x, 0)
        else:
            return x
