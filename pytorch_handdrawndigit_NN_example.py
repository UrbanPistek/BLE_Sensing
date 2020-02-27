import torch
import torchvision #contains a bunch of vision data
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''
************Torch Basics*********************
#Multipling two tensors
x = torch.tensor([5,3])
y = torch.tensor([2,1])

# print(x*y)

x= torch.zeros([2,5]) #Makes it all zeros

print(x)

tensor([[0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]])

y = torch.rand([2,5]) #random tensor
y = y.view([1,10]) #reshapes the matrix into a vector with 10 elements - "flattens it" 

'''

#***************** Data ***************
'''
You want two different datasets, the training and then the testing datasets
The testing dataset must be out of sample - must not be included in the training dataset
'''
'''
#MNIST - Hand drawn dataset of numbers 0-9
train = datasets.MNIST("", train=True, download=True,
                       transform = transforms.Compose([transforms.ToTensor()])) #Fetch the data and transform it to a tensor, for training

test = datasets.MNIST("", train=False, download=True,
                       transform = transforms.Compose([transforms.ToTensor()])) #Fetch the data and transform it to a tensor, for testing

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True) #batch size is for fitting through samples of data -
# important when you have massive datasizes and can't fit it in your ram/gpu/cpu
#Also, batch size helps with making the NN weights be more generalized
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True) #Shuffle randomizes how the data is being feed in to help with generalizations
#Typically you will have to batch and shuffle the data on your own
'''
'''
#Iterating over Data
for data in trainset:
    print(data)
    break

#Note Data is an object that contians a tensor of tensors that are your images and a tensor of tensors that are your labels
x, y = data[0][0], data[1][0]

print(y) # Accessing the 2nd index, which contains a tensor of tensors, then accessing the 0th tensor of those tensors
'''

'''
To view the first image 
plt.imshow(data[0][0].view(28,28)) #Transform to a 28x28 grid to be viewed
plt.show() #Show the first image
'''

'''
#Another Note; You need to ensure your dataset is balanced - your data in comprised of eqaul amount of the different "values"
#Interate through to determine how much of each type is in your data
total = 0;
counter_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

#Iterate through
for data in trainset:
    Xs, Ys = data
    for y in Ys:
        counter_dict[int(y)] += 1
        total += 1

print(counter_dict)

#Output percentages 
for idx in counter_dict:
    print(f"{idx}:{counter_dict[idx]/total*100}")
'''

#Building the Neural Network
train = datasets.MNIST("", train=True, download=True,
                       transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True,
                       transform = transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

class Net(nn.Module):

    def __init__(self):
        super().__init__() #Runs the initialization for nn and whatever else you pass into __init__
        self.fc1 = nn.Linear(784, 64)#input, output (input needs to be flattened), target is to make 3x64 hidden layers
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):#Defining how data will pass through
        x = F.relu(self.fc1(x))#Passing x through all the layers
        x = F.relu(self.fc2(x))#F.relu is the activation function (typically a sigmoud function) relu - rectified linear
        x = F.relu(self.fc3(x))
        x = self.fc4(x) #Here we want a function that gives a probablilty distribution, something that will indicate a clear output
        #Use log soft max
        return F.log_softmax(x, dim=1)

'''
#Can define the NN
#net = Net()
#print(net)

#Note on passing Data through
X = torch.rand((28,28))
X = X.view(-1, 28*28) #-1 Indicates that any sized tensor can be passed to be flattened 
'''
net = Net()

#Loss: A measure of how wrong the model is

optimizer = optim.Adam(net.parameters(), lr=0.001) #Calculate loss an feed information back into the model
#The lr = learning rate, this is how big of step sizes the NN takes to solve the problem, in typical problems you will want to use a decaying learning rate

EPOCHS = 3 #This is how many times you iterate through the entire dataset

for epoch in range(EPOCHS):
    for data in trainset:
        #data is a batch of featuresets and labels
        X, Y = data
        net.zero_grad() #Gradients contain the loss for you network to help optimize the network
        output = net(X.view(-1,28*28))
        loss = F.nll_loss(output, Y) #If your output result is a singular scalar value, use nll_loss
        loss.backward() #Backpropagating the loss
        optimizer.step() #This will optimize the weights for us
    print(loss)


#************Validating the Data to See How Correct it is***************
correct = 0
total = 0

with torch.no_grad(): #We don't wanna optimize with this data, just want to see how correct we were
    for data in trainset:
        X, y = data
        output = net(X.view(-1, 784))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 3))

#Verify accuracy
plt.imshow(X[3].view(28,28))
plt.show()
print(torch.argmax(net(X[3].view(-1, 784))[0]))

