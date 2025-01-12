#import torch
from torch import nn

class ButterflyHybridDetector(nn.Module):
    #import trained layers from ButterflyClassDetector.
    #add more layers on top and finetune.
    
    """Initializer method. Specifies layers as `self.` attributes."""
    def __init__(self):
        super().__init__()

        #should have a way to import the pretrained 14-class subspecies classifier with frozen weights

        ###classifying hybrid/nonhybrid with 2 layers
        self.hybrid_layers = nn.Sequential(
            #Layer 1: hidden layer that creates combos
            #of the 14 subspecies class probabilities
            nn.Linear(14, 14 ** 2),
            nn.ReLU(),
            #Layer 2: output layer that turns combos into
            #hybrid/nonhybrid logits
            nn.Linear(14 ** 2, 2)
            #no need to ReLU these, will softmax them
            #in train/evail
        )

    def forward(self, x):
        #transform x into 14-class logit vectors with 
        #the imported ButterflyClassDetector
        subspecies_logits = 

        #pass inputs into hybrid_layers
        hybrid_logits = self.hybrid_layers(subspecies_logits)

        return hybrid_logits