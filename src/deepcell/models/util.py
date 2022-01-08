from typing import List

import torch.nn


def truncate_model_to_layer(model: torch.nn.Module, layer: int) -> List:
    """Truncates model to layer

    Args:
        model: Model to truncate
        layer: Layer to truncate to

    Returns:
        List of layers truncated to layer
    """
    conv_layers = list(model.children())[0]
    return conv_layers[:layer]


def get_last_filter_num(layers: torch.nn.Sequential) -> int:
    """
    Gets the number of filters in the last layer of the CNN
    Args:
        layers: The CNN layers

    Returns:
        Number of filters in last layer of CNN

    Raises
        RuntimeError if this search fails
    """
    idx = -1
    while idx > -1 * len(layers):
        if hasattr(layers[idx], 'out_channels'):
            return layers[idx].out_channels
        idx -= 1

    raise RuntimeError('Could not find number of filters in last conv layer')
