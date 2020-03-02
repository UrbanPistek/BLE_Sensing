import keras
from keras.utils import np_utils
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
from scipy import stats
import os
#Optimize to run on the GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Just disables the warning, doesn't enable AVX/FMA

REBUILD_DATA = True

if torch.cuda.is_available(): #Check to see if GPU is availible to be used
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

EMPTY = 0
ONE_PERSON = 1
TWO_PLUS_PEOPLE = 2
LABELS = [EMPTY, ONE_PERSON, TWO_PLUS_PEOPLE] #DNN needs numeric labels
TIME_STEPS = 50
TRAIN_LEN = 60 #Train Dataset length (number of time matrices)
TEST_LEN = 12 #Test Dataset length (number of time matrices)
NUM_PARAMETERS = 4 #Number of parameters per timestep

def load_dataset():
    data = pd.read_csv('dataset1.csv')
    # Normalize the Data - Note Data is all negative so dividing by istelf twice will make it positive
    data['B1'] = data['B1'] / data['B1'].min()
    data['B2'] = data['B2'] / data['B2'].min()
    data['B3'] = data['B3'] / data['B3'].min()
    data['B4'] = data['B4'] / data['B4'].min()

    data['Label'].replace({"Empty": 0, "One person": 1, "Two people": 2}, inplace=True) #Replace Strings with values

    # Round numbers
    data = data.round({'B1': 6, 'B2': 6, 'B3': 6, 'B4': 6})

    return data

def make_traing_data(df, time_steps, train_len):
    N_FEATURES = 4

    segments = []
    labels = []
    for idx in tqdm(range(train_len)):
        for i in range(time_steps*idx, time_steps*(idx+1)): #Verify
            B1 = df['B1'].values[i: i + time_steps]
            B2 = df['B2'].values[i: i + time_steps]
            B3 = df['B3'].values[i: i + time_steps]
            B4 = df['B4'].values[i: i + time_steps]
            # Retrieve the most often used label in this segment
            label = stats.mode(df['Label'][i: i + time_steps])[0][0] #modify label_name
            segments.append([B1, B2, B3, B4])
            labels.append(label)

            # Bring the segments into a better shape
            reshaped_segments = np.asarray(segments, dtype=np.float64).reshape(-1, time_steps, N_FEATURES) #Modify
            value = np.asarray(labels) #Modify

    return reshaped_segments, value

#Pandas working
data = load_dataset()
print(data.head(5))
#print(len(data)) #3666

X, y = make_traing_data(data, TIME_STEPS, 6)
INPUT_SHAPE = NUM_PARAMETERS*TIME_STEPS
NUM_RESULTS = 3

def prep_data(X, y):
    train_X = X[0:250]
    train_y = y[0:250]
    test_X = X[251:]
    test_y = y[251:]

    print("Training X: ", train_X)
    print("Training y: ",train_y)
    print("Length of Train_X: ",len(train_X))
    print("Length of Train_y: ",len(train_y))
    print("Test X: ", test_X)
    print("Test y: ",test_y)
    print("Length of test_X: ",len(test_X))
    print("Length of test_y: ",len(test_y))

    print("******************* Training Data Shape *******************")
    print('x_train shape: ', train_X.shape)
    print(train_X.shape[0], 'training samples')
    print('y_train shape: ', train_y.shape)

    print('train_X shape: ', test_X.shape)
    print('train_y shape: ', test_y.shape)

    #================= Flattening train_X to Pass into the NN ========================
    print("=========================== Flattening and Inputting Data ==========================")
    train_X = train_X.reshape(train_X.shape[0], INPUT_SHAPE) #Flatten Training Data
    test_X = test_X.reshape(test_X.shape[0], INPUT_SHAPE) #Flatten Testing Data
    print('train_X shape:', train_X.shape)
    print('test_X shape:', test_X.shape)

    train_X = train_X.astype('float32')
    train_y= train_y.astype('float32')
    test_X = test_X.astype('float32')
    test_y= test_y.astype('float32')

    train_y = np_utils.to_categorical(train_y, NUM_RESULTS)
    test_y = np_utils.to_categorical(test_y, NUM_RESULTS)
    print('train_y shape:', train_y.shape)
    print('test_y shape:', test_y.shape)

    return train_X, train_y, test_X, test_y

train_X, train_y, test_X, test_y = prep_data(X, y)

# ============================= Add Pytorch Implementation Here =====================================
