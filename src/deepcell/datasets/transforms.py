import torch
import torchvision.transforms.functional as TF
import random


class RandomRotate90:
    """Rotates by a random multiple of 90 degrees"""
    def __call__(self, x: torch.tensor):
        n_rotatations = random.choice(range(-3, 4))
        angle = 90 * n_rotatations
        return TF.rotate(x, angle)
