#import torch
from torch import nn

class ButterflyClassDetector(nn.Module):
    #start with a few layers. follow a pytorch tutorial.
    #check islp notes: pooling->conv->pooling. reduce scramble reduce. turn it to 14 classes and train on cross
    #entropy. *even if it's not very good,* freeze & import into butterfly_hybrid_detector for next stage.
    
    """Initializer method. Specifies layers as `self.` attributes of class object."""
    def __init__(self):
        super().__init__()

    """Feed forward method. Receives data input vector X and uses layers (as attributes specified in __init__) to transform it into 14 class response logits."""
    def forward(self, x):
        #question: I think we should avoid flattening as the first thing we do to the image. Instead, preserve 2D structure so we can do Conv2d (look for features over 2D space) and Pooling 2D, then flatten at the end.
        
        #example:
        #obtain 1D vector from original observation
        #x_vector = self.flatten(x) 
        #pass inputs into 4 layer neural network
        #response_logits = self.linear_layer_stack(x_vector)
        return response_logits #length 14
        