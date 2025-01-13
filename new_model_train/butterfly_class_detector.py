#import torch
from torch import nn

class ButterflyClassDetector(nn.Module):
    #start with a few layers. follow a pytorch tutorial.
    #check islp notes: pooling->conv->pooling. reduce scramble reduce. turn it to 14 classes and train on cross
    #entropy. *even if it's not very good,* freeze & import into butterfly_hybrid_detector for next stage.
    
    """Initializer method. Specifies layers as `self.` attributes of class object."""
    def __init__(self, num_classes = 2):
        super().__init__()

        #convolution and padding layers
        #each conv increases channels/depth from 3 to 16 to 32 to 64
        #each pool halves length and width from 224 to 112 to 56 to 28.
        self.conv_pad_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2,2)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(64 * 28 * 28, 512),
            nn.ReLU(),
            #nn.Dropout(0.5)
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 14)  #14 logits
        )

        self.flatten = nn.Flatten()    

    """Feed forward method. Receives data input vector X and uses layers (as attributes specified in __init__) to transform it into 14 class response logits."""
    def forward(self, x):
        x_convoluted = self.conv_pad_layers(x)
        x_vector = self.flatten(x_convoluted) 
        response_logits = self.linear_layers(x_vector)
        return response_logits