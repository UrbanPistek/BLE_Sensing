'''
Typically, convolutional neural networks were used for image tasks
Recently, convolutional neural networks appear to be out-preforming recurrent neural networks for sequntial types of data

Convolutional NN:
Accepts both 2D and 3D data as inputs, therefore the input does not need to be flatened
How it works in image processing is that it takes in your image, then using a "window" of some grid size (3px by 3px) - referred
to as kernals, it scans over the image and looks for features like edges. You are left with a simplified image which then using a process
called "pooling" it find the largest value in your kernal and makes a more simplified image with that.

In general, it takes your image and makes a more simplified image
'''

import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REBUILD_DATA = False #Flag for pre-processing step, typically you will want a seperate program for preprocessing data
#This way the training dataset is only run/created when you choose

class DogsVsCats():
    IMG_SIZE = 50 #Need to make all images from the dataset uniform
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []
    catcount = 0
    dogcount = 0 #Pay attention to balance

    def make_training_data(self):
        for label in self.LABELS: #Iterate over the label
            for f in tqdm(os.listdir(label)): #Iterating over all the files in the directory
                if "jpg" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) #Always try to simplify the incoming data - what is neccessary - and minimize the size of the NN
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])  #To make one-hot-vector: np.eye(10)[7] - vector of size 10 with the 7th index being 1

                        if label == self.CATS:
                            self.catcount += 1
                        elif label == self.DOGS:
                            self.dogcount += 1
                        #Check ot see that both have almost the same amount of samples
                    except Exception as e:
                        pass
                        #print(str(e))

        #After these operations the training data will be mixed with cat and dog images with their associated labels
        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("Cats:", self.catcount)
        print("Dogs:", self.dogcount)

if REBUILD_DATA:
    dogsvcats = DogsVsCats()
    dogsvcats.make_training_data()

training_data = np.load("training_data.npy", allow_pickle=True)
# ^ To load back in the training data

'''
#Check that the dataset has been properly created 
plt.imshow(training_data[1][0], cmap="gray")
plt.show()
'''

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #Creating the Layers
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        #With this library there is no straightforward method to flatten the information to a Linear Layer, need to use a guess and check strategy
        #Essentially, pass random data through and check the size to determine the flatten operation needed
        x = torch.rand(50,50).view(-1,1,50,50) #Generating the random data
        self._to_linear = None
        self.convs(x)

        # Need to pass information to a linear layer to get the final output
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x): #Only need to run once to determine the number to flatten to
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x) #pass it through all the convolutional layers
        x = x.view(-1, self._to_linear) #Flatten
        x = F.relu(self.fc1(x)) #Pass through first linear layer
        x = self.fc2(x) #Passing it through the final fully connected layer
        return F.softmax(x, dim=1) #Good to have an activation function on the last output

net = Net()

#***************Training the Network***********************

#Seperating out the testing and training data
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss() #Loss using mean squared error

#Seperate out the X's and the Y's
X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

#Seperating out Training and testing Data
VAL_PCT = 0.1 #Testing against 10% of our dataset
val_size = int(len(X)*VAL_PCT)

train_X = X[:-val_size] #python slicing
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

#Traing the Network
BATCH_SIZE = 100 #If you have an memory error typically lower the batch size
EPOCHS = 5
def train(net):
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)): #Slices in the training data, where the first batch will be the zeroth to 100th index and so-fourth
            batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,50,50)
            batch_y = train_y[i:i + BATCH_SIZE]

            net.zero_grad()
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()

    print(loss)

#******************Predicting on the Model******************
def test(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax((test_y[i]))
            net_out = net(test_X[i].view(-1,1, 50, 50))[0]
            predicted_class = torch.argmax(net_out)
            if predicted_class == real_class:
                correct += 1
            total += 1

    print("Accuracy: ", round(correct/total, 3))

#****************** Running on the GPU ****************
#print(torch.cuda.is_available()) #Test to see is CUDA is availible
#print(torch.device_count()) #Check to see how many GPUs are availible, therefore you can run different models on different GPUs

if torch.cuda.is_available(): #Check to see if GPU is availible to be used
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

#Note:
'''
There is a conversion happening when you move things to the GPU, therefore you either move your entire dataset to the GPU or move certain processes to the GPU 
'''

net.to(device)

def train_GPU(net): #Training the model on the GPU
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):  # Slices in the training data, where the first batch will be the zeroth to 100th index and so-fourth
            batch_X = train_X[i:i + BATCH_SIZE].view(-1, 1, 50, 50)
            batch_y = train_y[i:i + BATCH_SIZE]

            batch_X , batch_y = batch_X.to(device), batch_y.to(device)

            net.zero_grad()
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()

    print(loss)

def test_GPU(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax((test_y[i])).to(device)
            net_out = net(test_X[i].view(-1, 1, 50, 50).to(device))[0] #returns a list
            predicted_class = torch.argmax(net_out)
            if predicted_class == real_class:
                correct += 1
            total += 1

    print("Accuracy: ", round(correct / total, 3))

#for i in range(5): #For training the model multiple times
    print("Iteration: ", i)
    train_GPU(net)
    test_GPU(net)
print("done")
'''
When training one thing to keep track of is whether loss is going down - this means the model is learning something 
The biggest thing is the difference between in-sample accuracy and out-sample accuracy 
'''


