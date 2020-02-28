'''
For recognizing the difference between Cats and Dogs images
By Modifying the Dataset and labels it should be easy to adapt it to any image recognition
'''

import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import pandas as pd

REBUILD_DATA = True

if torch.cuda.is_available(): #Check to see if GPU is availible to be used
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

class RSSI_Data():
    dataset = []
    testing_data = []
    training_data = []
    timesteps = 100
    EMPTY = "Empty"
    ONE_PERSON = "1_PERSON"
    TWO_PERSONS = "2_PERSONS"
    LABELS = {EMPTY: 0, ONE_PERSON: 1, TWO_PERSONS: 2}
    empty_data_count = 0
    one_person_data_count = 0
    two_person_data_count = 0

    data = pd.DataFrame() #Possibly not necessary

    def make_dataset(self):
        data = pd.read_csv('1D_CNN_Dataset1.csv')

#Look into making training and testing data has functions later

#Pandas working 
data = pd.read_csv('1D_CNN_Dataset1.csv')
print(data)