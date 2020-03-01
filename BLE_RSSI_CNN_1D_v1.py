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
from scipy import stats

REBUILD_DATA = True

if torch.cuda.is_available(): #Check to see if GPU is availible to be used
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

LABELS = {'Empty': 0,'One person': 1, 'Two or more people':2} #DNN needs numeric labels
TIME_STEPS = 50
STEP_INCREMET = 1

def load_dataset():
    data = pd.read_csv('dataset1.csv')

def make_traing_data(df, time_steps, step):
    N_FEATURES = 4

    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step): #Verify
        B1 = df['B1'].values[i: i + time_steps]
        B2 = df['B2'].values[i: i + time_steps]
        B3 = df['B3'].values[i: i + time_steps]
        B4 = df['B4'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        '''label = stats.mode(df[label_name][i: i + time_steps])[0][0] #modify label_name'''
        segments.append([B1, B2, B3, B4])
        labels.append(label)

        # Bring the segments into a better shape
        reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, N_FEATURES) #Modify
        labels = np.asarray(labels) #Modify

    return reshaped_segments, labels

#Pandas working 
data = pd.read_csv('dataset1.csv')
columns = data.columns
print(data.head(5))
print(len(data)) #3666