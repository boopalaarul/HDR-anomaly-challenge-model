#import torch
from torch import nn
from butterfly_class_detector import ButterflyClassDetector

device = "cuda:0"

"""HybridDetector needs to incorporate ClassifierDetector as a base model, since it is being fed images.""".
class ButterflyHybridDetector(nn.Module):
    #import trained layers from ButterflyClassDetector.
    #add more layers on top and finetune.
    
    """Initializer method. Specifies layers as `self.` attributes."""
    def __init__(self):
        super().__init__()

        ###import the pretrained 14-class subspecies classifier with frozen weights
        self.base_model = ButterflyClassDetector.to(device)
        self.base_model.load_state_dict(
            torch.load("ButterflyMNISTClassifier_StateDict.pth"), 
            weights_only = True,
            map_location=torch.device(device)
        )
        for param in self.base_model.parameters():
            param.requires_grad = False
            
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
        subspecies_logits = self.base_model(x)

        #pass inputs into hybrid_layers
        hybrid_logits = self.hybrid_layers(subspecies_logits)

        return hybrid_logits