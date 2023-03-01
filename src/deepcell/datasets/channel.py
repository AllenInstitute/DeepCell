from enum import Enum


class Channel(Enum):
    """A single channel of `RoiDataset`"""
    MASK = 'MASK'
    MAX_PROJECTION = 'MAX_PROJECTION'
    AVG_PROJECTION = 'AVG_PROJECTION'
    CORRELATION_PROJECTION = 'CORRELATION_PROJECTION'
    MAX_ACTIVATION = 'MAX_ACTIVATION'


# maps channel type to expected filename prefix,
# e.g. the mask channel has name like mask_0_0.png
channel_filename_prefix_map = {
    Channel.MASK: 'mask',
    Channel.MAX_PROJECTION: 'max',
    Channel.AVG_PROJECTION: 'avg',
    Channel.CORRELATION_PROJECTION: 'correlation',
    Channel.MAX_ACTIVATION: 'max_activation'
}
