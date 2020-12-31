from torch import nn

from models.video.RNNDecoder import RNNDecoder
from models.video.CNNEncoder import CNNEncoder


class VideoNet(nn.Module):
    def __init__(self, cnn: CNNEncoder, rnn: RNNDecoder):
        super().__init__()

        self.cnn = cnn
        self.rnn = rnn

    def forward(self, x):
        x = self.cnn(x)
        x = self.rnn(x)
        return x
