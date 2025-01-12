import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
#turns image files into nonflattened 2d vectors? but isn't the data
#already in binary, over in data/...
from torchvision.transforms import ToTensor
from fashion_mnist_classifier import FashionMNISTClassifier

#everything in this library is about inheriting from classes to create objects,
#then instantiating those objects.
#if you want to load a particular model's state dict, have to load it into an 
#object of its *own class*

device = "cpu"#"cuda"
#model is instance of class that extends nn.Module.
#shouldn't inherently belong to any device until set to one?
model = FashionMNISTClassifier().to(device) 
model.load_state_dict(torch.load("FashionMNISTClassifier_StateDict.pth",
                                    weights_only = True,
                                    map_location=torch.device("cpu")))
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

#this is what prints those training style "epoch" messages - just so we know
#how our model did when we trained it, what to expect
model.eval()

data_test = FashionMNIST(
    root = "data",
    train = False,
    download = False, #already downloaded
    transform = ToTensor()    
)

#no need for dataloader because there's no need for batching: no training.
#batch_size = 64
#dataloader_test = DataLoader(data_test, batch_size = batch_size)

#this must be a mappable dataset: use indices to get particular samples
x, y = data_test[0][0], data_test[0][1]
print(f'Observation shape {x.shape}, label {y}')
with torch.no_grad():
    x = x.to(device) #make this a CUDA tensor
    pred = model(x)
    print(pred)
    #y is a numerical label, which corresponds to a certain word
    #label obtained from pred by argmax over pred[0]... pred assumes a
    #tensor of logit values, so a vector of vectors, but there is only one 
    #sample in x.
    #even if this is the case! for some reason pred.item() doesn't work.
    #so pred.item() can only even return a scalar, from 1x1x1... tensor
    predicted_class, actual_class = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: {predicted_class}, Actual: {actual_class}')
