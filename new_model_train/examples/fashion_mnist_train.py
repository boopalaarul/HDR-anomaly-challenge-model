#this is an OOP library. you import base classes, create objects that
#instantiate & inherit them while adding new properties

#provides SGD function, data types, and option contexts (torch.no_grad())
import torch

#import model class created earlier, as well as loss function from library
from fashion_mnist_classifier import FashionMNISTClassifier
from torch.nn import CrossEntropyLoss()

#dataset methods. torchvision already contains the (inheritor) of Dataset 
#class, FashionMNIST. Don't have to import generic Dataset class.
from torch.utils.data import DataLoader#, Dataset
import torchvision as tv

#from torchvision import datasets
#from torchvision.transforms import ToTensor

print(f"Pytorch version {torch.__version__}")

"""Data import -- these constructors will download the data locally while 
creating variables that refer to them."""
data_train = tv.datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = tv.transforms.ToTensor()
) 
data_test = tv.datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = tv.transforms.ToTensor()
) 

"""Data preparation -- specifying batch size for training"""
batch_size = 64
#These vars are tuples (X, y).
dataloader_train = DataLoader(data_train, batch_size = batch_size)
dataloader_test = DataLoader(data_test, batch_size = batch_size)

print(f"Number of batches (size <= 64) in training data: {len(dataloader_train)}")
for X, y in dataloader_train:
    print(f"Shape of X[N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break #only want to see shape of first batch object
print(f"Number of batches (size <= 64) in test data: {len(dataloader_test)}")
for X, y in dataloader_test:
    print(f"Shape of X[N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

"""Creating neural network -- already done in another module, class imported"""

"""Specifying device for model"""
device = (
    "cuda"
    #if torch.cuda.is_available()
    #else "mps"
    #if torch.backends.mps.is_available()
    #else "cpu"
)
model = FashionMNISTClassifier().to(device)
print(model)

"""Specifying loss function and "optimizer"-- optimizer is what uses the loss function, using its gradient (contains residual) to adjust parameters according to learning rate: a different "optimizer" might be used to improve the learning rate, number of layers, and number of neurons per layer"""

#as expected for classification problem where outputs are logits
loss_function = CrossEntropyLoss()
#stochastic gradient descent, starting in a random place and adjusting params
#(the 3 bias-weight matrices of linear layer stack) to reach minimum in
#loss fn gradient. observations are fed in batches of 64, so every 64 obs
#the function tries somewhere else
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

"""Routine for training model - earlier was more object oriented, now it's like
the 'main method'"""
#dataloader_train is fed into this, "model" combines FashionMNISTClassifier
#and its device
def train(dataloader, model, loss_function, optimizer):
    size = len(dataloader.dataset)
    model.train()
    #for each batch: "batch" is likely a batch ID, X & y are the data
    for batch, (X, y) in enumerate(dataloader): #iterable to iterator

        #dataset objects (are still 28x28 matrices & labels) are tensors,
        #vectors which are associated to a particular device & datatype with
        #"to()" method
        X, y = X.to(device), y.to(device)

        #prediction error - model, used as a method, is like ".fit_transform()"
        pred = model(X)
        #loss function class returns a vector (tensor) of loss values for
        #each observation in batch...? but Tensor docs say item() = singleton.
        loss = loss_function(pred, y)

        #backpropagation
        loss.backward() #for any tensor, backward() finds negative gradient
        optimizer.step() #mult by learning rate, adj.model by reference?
        optimizer.zero_grad() #what...?
        
        if batch % 100 == 0: #if this is the 100th batch of training
            #get the loss function value for the most recent prediction?
            loss, current = loss.item(), (batch+1) * len(X)
            #format string to print loss up to 7 decimal pts
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

"""Defining test routine"""

#calculate test error on dataloader_test
def test(dataloader, model, loss_function):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    #what is this doing to model???
    model.eval()
    
    #starting off at 0 each
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device) 
            pred = model(X)
            test_loss += loss_function(pred, y).item()
            #loss doesn't correspond to wrongness, but disunity. so we
            #find the "actual prediction" & evaluate correctness as "is
            #the class w/ biggest logit right or wrong"
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

"""main method: specifying epochs for training."""

epochs = 5
for t in range(epochs):
    print(f"EPOCH {t+1}\n==================================")
    train(dataloader_train, model, loss_function, optimizer)
    test(dataloader_test, model, loss_function)
print("Done.")


"""saving model -- can recreate model from a serialized dictionary of its
internal state: parameters and so on"""
torch.save(model.state_dict(), "FashionMNISTClassifier_StateDict.pth")
