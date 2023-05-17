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
            return torch.flip(x, dims=(0,))
        else:
            return x


class RandomRollVideo:
    """Randomly chooses a starting index and when it gets to the end,
    rolls the beginning"""
    def __init__(self, p):
        self._p = p

    def __call__(self, x: torch.tensor):
        if random.random() < self._p:
            shift = random.choice(range(len(x)))
            return torch.roll(x, shifts=shift, dims=0)
        else:
            return x


class RandomClip:
    """Randomly selects a clip of len"""
    def __init__(self, len: int):
        self._len = len

    def __call__(self, x: torch.tensor):
        if self._len > x.shape[0]:
            raise ValueError(f'len {self._len} must be less than n frames in '
                             f'array')
        start = random.choice(range(len(x) - self._len + 1))
        return x[start:start+self._len]


class SubselectClip:
    """Selects a clip of length len from the input"""
    def __init__(self, len: int, start_idx: int):
        self._len = len
        self._start_idx = start_idx

    def __call__(self, x: torch.tensor):
        if self._len > x.shape[0]:
            raise ValueError(f'len {self._len} must be less than n frames in '
                             f'array')
        x = x[self._start_idx:self._start_idx + self._len]

        # if length of array too short
        if len(x) < self._len:
            # take frames from beginning
            wrap_len = self._len - len(x)
            x = torch.concatenate([x[:wrap_len], x])

        return x


class ReduceFrameRate:
    """Downsample video in time"""
    def __init__(self, temporal_downsampling: int = 6):
        self._temporal_downsampling = temporal_downsampling

    def __call__(self, x: torch.tensor):
        return x[torch.arange(0, len(x), self._temporal_downsampling)]
