import torch
from torch import nn

class ButterflyClassDetector(nn.Module):
    #start with a few layers. follow a pytorch tutorial.
    #check islp notes: pooling->conv->pooling. reduce scramble reduce. turn it to 14 classes and train on cross
    #entropy. *even if it's not very good,* freeze & import into butterfly_hybrid_detector for next stage.