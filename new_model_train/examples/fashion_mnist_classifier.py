#this is an OOP library. you import base classes, create objects that
#instantiate & inherit them while adding new properties

#don't need any classes besides this right now: we only want to make a
#neural network object extending nn.Module. is agnostic to device.
#data matters only for the layer sizes: input has to be 28x28 objects
#in order for Flatten() to produce correct length tensor (or 14x56...)

from torch import nn

"""Creating neural network -- specify tunable parameters of model"""

class FashionMNISTClassifier(nn.Module):
    def __init__(self):
        #initiate the base class for shared neural network methods/vars
        super().__init__()
        #unique vars of this object: layers & neurons per layer.
        #Layer 0: Flattening 28x28 image into 1D vector, length 28^2.
        self.flatten = nn.Flatten()
        #Layers 1-4: stack of neuron layers, each neuron is linear combo
        #of the ones that feed into it according to learned biases/weights,
        #followed by activation function... layers don't have to be linear,
        #can be recurrent/autoregressive, or dropouts w random zero-outs.
        #nn.Sequential is a container for sequential layers 
        self.linear_layer_stack = nn.Sequential(
            #Layer 1 to 2: going from 28^2 neurons to just 512
            nn.Linear(28 ** 2, 512),
            #Linear transform followed by ReLU = overall nonlinear transform
            nn.ReLU(),
            
            #Layer 2 to Layer 3
            nn.Linear(512, 512),
            nn.ReLU(),
            
            #Layer 3 to final layer of 10 neurons, each representing
            #probability of one of 10 response classes, minimizing
            #crossentropy with a regularization term for loss function (?)
            nn.Linear(512, 10)
        )
    #forward() defines how data actually passes through network: so can set
    #up different kinds of networks in init (different containers like Module
    #List or ModuleDict) and then define the control mechanisms for which
    #goes to which here (taking in multimedia input...?)
    def forward(self, x):
        #obtain 1D vector from original observation
        x_vector = self.flatten(x) 
        #pass inputs into 4 layer neural network
        response_logits = self.linear_layer_stack(x_vector)
        return response_logits
