import os
from os.path import dirname, abspath, join
import torch


CODEPATH = dirname(abspath(__file__))
DATAPATH = abspath(join(join(CODEPATH, os.pardir), 'data'))
RESPATH = abspath(join(join(CODEPATH, os.pardir), 'results'))


Tensor = torch.DoubleTensor
