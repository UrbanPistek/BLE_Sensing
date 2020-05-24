from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.utils import np_utils
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import os
import tensorflow as tf
import random
import h5py

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
TIME_STEPS = 200 #Originally 50 #optimized 300
STEP_DISTANCE = 25 #If equal to TIME_STEPS there is no overlap in data #Originally 50 #optimized 25
TOTAL_LEN = 144 #Originally 72
TRAIN_LEN = 100 #Train Dataset length (number of time matrices) #Originally 60
TEST_LEN = 44 #Test Dataset length (number of time matrices) #Originally 12
NUM_PARAMETERS = 4 #Number of parameters per timestep
N_FEATURES = 4
DATASET = "dataset2"
MODEL = "model2_ker100_val0.3_opt"

def load_dataset():
    data = pd.read_csv('dataset2.csv') #optimize, best dataset = dataset2
    # Normalize the Data - Note Data is all negative so dividing by istelf twice will make it positive
    data['B1'] = data['B1'] / data['B1'].min()
    data['B2'] = data['B2'] / data['B2'].min()
    data['B3'] = data['B3'] / data['B3'].min()
    data['B4'] = data['B4'] / data['B4'].min()

    data['Label'].replace({"Empty": 0, "One person": 1, "Two people": 2}, inplace=True) #Replace Strings with values

    # Round numbers
    data = data.round({'B1': 6, 'B2': 6, 'B3': 6, 'B4': 6})

    return data

def make_training_data_tensor(df, time_steps, train_len): #returns a list of randomly shuffled [matrix, label] elements
    training_data = []
    result = []
    data = []
    for idx in tqdm(range(train_len)):
        for i in range(time_steps): #Verify
            B1 = df['B1'][(idx*time_steps) + i] #idx*time_steps + i
            B2 = df['B2'][(idx*time_steps) + i]
            B3 = df['B3'][(idx*time_steps) + i]
            B4 = df['B4'][(idx*time_steps) + i]
            print("Iteration: ", (idx*time_steps) + i, "[",
                  df['B1'][(idx*time_steps) + i],
                  df['B1'][(idx*time_steps) + i],
                  df['B1'][(idx*time_steps) + i],
                  df['B1'][(idx*time_steps) + i] ,"]",
                "_",df['Label'][(idx*time_steps) + i]) #For debugging
            data.append([B1, B2, B3, B4])

        print("STATS MODE VALE: ", stats.mode(df['Label'][(idx*time_steps): (idx*time_steps) + time_steps], axis=None)[0][0]) # For Debugging
        label = stats.mode(df['Label'][(idx*time_steps): (idx*time_steps) + time_steps], axis=None)[0][0]
        if label == 0:
            result = [1, 0, 0]
        elif label == 1:
            result = [0, 1, 0]
        elif label == 2:
            result = [0, 0, 1]

        training_data.append([data, result])
        np.random.shuffle(training_data)

    return training_data


def slice_data(testing):
    X = []
    Y = []
    for i in range(len(testing)):
        X.append(testing[i][0])
        Y.append(testing[i][1])

    test_X = np.asarray(X, dtype=np.float).reshape(-1, TIME_STEPS, NUM_PARAMETERS)
    test_Y = np.asarray(Y)
    return test_X, test_Y


def make_training_data(df, time_steps, step):
    segments = []
    labels = []
    #added to(device) to see if preformance improves
    for i in tqdm(range(0, len(df) - time_steps, step)):
        B1 = df['B1'].values[i: i + time_steps]
        B2 = df['B2'].values[i: i + time_steps]
        B3 = df['B3'].values[i: i + time_steps]
        B4 = df['B4'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        label = stats.mode(df['Label'][i: i + time_steps])[0][0]
        segments.append([B1, B2, B3, B4])
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype=np.float64).reshape(-1, time_steps, N_FEATURES) #Modify
    value = np.asarray(labels) #Modify

    return reshaped_segments, value

data = load_dataset()
#print(len(data)) #3666
#data.to_pickle("rssi_dataset1.pkl")
#data = pd.read_pickle("rssi_dataset1.pkl") #Read from pickle

X, y = make_training_data(data, TIME_STEPS, STEP_DISTANCE)
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Checking Data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("Input Data", X)
print("Result Data",y)

INPUT_SHAPE = NUM_PARAMETERS*TIME_STEPS
NUM_RESULTS = 3

def prep_data(X, y):
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Preparing Data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    train_X = X[0:len(X)]
    train_y = y[0:len(y)]

    print("Training X: ", train_X)
    print("Training y: ",train_y)
    print("Length of Train_X: ",len(train_X))
    print("Length of Train_y: ",len(train_y))

    print("******************* Training Data Shape *******************")
    print('x_train shape: ', train_X.shape)
    print(train_X.shape[0], 'training samples')
    print('y_train shape: ', train_y.shape)

    #================= Flattening train_X to Pass into the NN ========================
    print("=========================== Flattening and Inputting Data ==========================")
    train_X = train_X.reshape(train_X.shape[0], INPUT_SHAPE) #Flatten Training Data
    print('train_X shape:', train_X.shape)

    train_X = train_X.astype('float32')
    train_y= train_y.astype('float32')

    train_y = np_utils.to_categorical(train_y, NUM_RESULTS) #Only run once
    print('train_y shape:', train_y.shape)

    return train_X, train_y

def shuffle_data(X, y):
    print("Object_Type_X: ", type(X))
    print("Object_Type_y: ", type(y))
    c = list(zip(X, y))
    random.shuffle(c)
    X, y = zip(*c)
    X = np.asarray(X)
    y = np.asarray(y)
    print("After Shuffle: ")
    print("Object_Type_X: ", type(X))
    print("Object_Type_y: ", type(y))
    return X, y

train_X, train_y = shuffle_data(X, y)
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Checking Training Data after Shuffle >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("Train X [0]: ", train_X[0])
print("Train y [0]: ", train_y[0])
print("Train X: ", train_X)
print("Train y: ", train_y)

train_X, train_y = prep_data(train_X, train_y)
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Checking Training Data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("Train X [0]: ", train_X[0])
print("Train y [0]: ", train_y[0])
print("Train X: ", train_X)
print("Train y: ", train_y)

def create_model():
    #modified from original
    model_c = Sequential()
    model_c.add(Reshape((TIME_STEPS, NUM_PARAMETERS), input_shape=(INPUT_SHAPE,)))
    model_c.add(Conv1D(100, 10, activation='relu', input_shape=(TIME_STEPS, NUM_PARAMETERS)))
    model_c.add(Conv1D(100, 100, activation='relu'))
    model_c.add(MaxPooling1D(3))
    model_c.add(Conv1D(160, 10, activation='relu'))
    model_c.add(Conv1D(160, 10, activation='relu'))
    model_c.add(GlobalAveragePooling1D())
    model_c.add(Dropout(0.5))
    #model_c.add(Flatten())
    #model_c.add(Dense(100, activation='relu'))
    model_c.add(Dense(NUM_RESULTS, activation='softmax'))
    print(model_c.summary())

    return model_c
# ================================ Training the Model =====================================
model_c = create_model()
callbacks_list = []

callbacks_list = [
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=200) #Change to monitor='accuracy' if errors
]

opt = keras.optimizers.Adam(learning_rate=0.0001) #optimize, best around 0.0001
model_c.compile(loss='categorical_crossentropy',
                optimizer= opt, metrics=['accuracy']) #Check different ways of calculating loss

# Hyper-parameters
BATCH_SIZE = 20 #optimize, BEST is 25
EPOCHS = 500 #optimize, best 600

# Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
history = model_c.fit(train_X,
                      train_y,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.4,
                      verbose=1) #Check that the validation split is not splitting the data in weird ways -> shuffle the data in batches of 50

#Saving the model to file for later
#model_c.save("modelTest.h5")
model_c.save("model_epochs"+str(EPOCHS)+"_batch"+str(BATCH_SIZE)+"_time"+str(TIME_STEPS)+"_step"+str(STEP_DISTANCE)+DATASET+MODEL+".h5")
print("Model is saved")

plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], 'r', label='Accuracy of training data')
plt.plot(history.history['val_accuracy'], 'b', label='Accuracy of validation data')
plt.plot(history.history['loss'], 'g', label='Loss of training data')
plt.plot(history.history['val_loss'], 'y', label='Loss of validation data')
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()

# Print confusion matrix for training data
y_pred_train = model_c.predict(train_X)
# Take the class with the highest probability from the train predictions
max_y_pred_train = np.argmax(y_pred_train, axis=1)
#print(classification_report(train_y, max_y_pred_train))