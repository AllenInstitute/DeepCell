import logging
import sys

from torch import nn

from models.video.RNNDecoder import RNNDecoder
from models.video.CNNEncoder import CNNEncoder

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(levelname)s:\t%(asctime)s\t%(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)

logger = logging.getLogger(__name__)

class VideoNet(nn.Module):
    def __init__(self, cnn: CNNEncoder, rnn: RNNDecoder):
        super().__init__()

        self.cnn = cnn
        self.rnn = rnn

    def forward(self, x):
        logger.info('CNN')
        x = self.cnn(x)
        logger.info('DONE CNN')

        logger.info('RNN')
        x = self.rnn(x)
        logger.info('DONE RNN')
        return x
