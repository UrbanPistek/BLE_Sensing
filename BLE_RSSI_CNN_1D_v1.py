from __future__ import print_function
import keras
import matplotlib
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt

import sklearn
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

import scipy
import seaborn as sns
from IPython.display import display, HTML

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
TOTAL_LEN = 72
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

def make_training_data_tensor(df, time_steps, train_len): #returns a list of randomly shuffled [matrix, label] elements
    N_FEATURES = 4
    training_data = []
    lab_val = []
    result = []
    data = []
    for idx in tqdm(range(train_len)):
        for i in range(time_steps): #Verify
            B1 = df['B1'][(idx*time_steps) + i] #idx*time_steps + i
            B2 = df['B2'][(idx*time_steps) + i]
            B3 = df['B3'][(idx*time_steps) + i]
            B4 = df['B4'][(idx*time_steps) + i]
            print("Iteration: ", (idx*time_steps) + i, "[", df['B1'][(idx*time_steps) + i],df['B1'][(idx*time_steps) + i],df['B1'][(idx*time_steps) + i],df['B1'][(idx*time_steps) + i] ,"]",
                "_",df['Label'][(idx*time_steps) + i]) #For debugging
            data.append(np.array([B1, B2, B3, B4]))

        print("STATS MODE VALE: ", stats.mode(df['Label'][(idx*time_steps): (idx*time_steps) + time_steps], axis=None)[0][0]) # For Debugging
        label = stats.mode(df['Label'][(idx*time_steps): (idx*time_steps) + time_steps], axis=None)[0][0]
        if label == 0:
            result = np.array([1, 0, 0])
        elif label == 1:
            result = np.array([0, 1, 0])
        elif label == 2:
            result = np.array([0, 0, 1])

        training_data.append([np.array(data), result])
        np.random.shuffle(training_data)

    return training_data

def make_training_data(df, time_steps, train_len):
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
            label = stats.mode(df['Label'][i: i + time_steps])[0][0] #STATS MODE WAS MAKING ALL THE RESULTS ZERO
            segments.append([B1, B2, B3, B4])
            labels.append(label) #Verify how the output is being produced

            # Bring the segments into a better shape
            reshaped_segments = np.asarray(segments, dtype=np.float64).reshape(-1, time_steps, N_FEATURES) #Modify
            value = np.asarray(labels) #Modify

    return reshaped_segments, value

#Pandas working 
data = load_dataset()
#print(len(data)) #3666
#data.to_pickle("rssi_dataset1.pkl")

#data = pd.read_pickle("rssi_dataset1.pkl") #Read from pickle
print(data.head(5))
print("Data Label Value Ranges")
print(data['Label'][0:3])
print(data['Label'][1500:1503])
print(data['Label'][3000:3003])

X, y = make_training_data(data, TIME_STEPS, TOTAL_LEN) #Create function to shuffle the data in chunks of TIME_STEPS
#print(X[0:3])
#print(y[0:3])

print("=================================================================")
testing = make_training_data_tensor(data, TIME_STEPS, TOTAL_LEN)
print("*********** TESTING *************")
print("Label:", data['Label'][0])
print(testing[0])
print("****************************************")
print("Label", data['Label'][3000])
print(len(testing))
print(testing[60])
print("****************************************")
print("Label", data['Label'][3200])
print(testing[64])
print("*********** TESTING *************")
for i in range(len(testing)):
    print(i, " : ", testing[i][1])

print("TESTING FOR TRAIN_X AND TRAIN_Y")

def slice_data(testing):
    test_X = []
    test_Y = []
    for i in range(len(testing)):
        test_X.append(testing[i][0])
        test_Y.append(testing[i][1])
    return test_X, test_Y

test_X, test_Y = slice_data(testing)

print("Inputs: ", test_X[0:3])
print("Outputs: ", test_Y[0:3])
print("==============================================")

INPUT_SHAPE = NUM_PARAMETERS*TIME_STEPS
NUM_RESULTS = 3

def prep_data(X, y):
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

train_X, train_y = prep_data(X, y)

''' =========================================== OLD =========================================
model_m = Sequential()
model_m.add(Reshape((TIME_STEPS, 4), input_shape=(INPUT_SHAPE,)))
model_m.add(Dense(100, activation='relu'))
model_m.add(Dense(100, activation='relu'))
model_m.add(Dense(100, activation='relu'))
model_m.add(Flatten())
model_m.add(Dense(NUM_RESULTS, activation='softmax'))
print(model_m.summary())
   =========================================== OLD ========================================= # Add comment to here

def create_model():
    model_c = Sequential()
    model_c.add(Reshape((TIME_STEPS, NUM_PARAMETERS), input_shape=(INPUT_SHAPE,)))
    model_c.add(Conv1D(64, 3, activation='relu', input_shape=(TIME_STEPS, NUM_PARAMETERS)))
    model_c.add(Conv1D(64, 3, activation='relu'))
    model_c.add(Dropout(0.5))
    model_c.add(MaxPooling1D(2))
    model_c.add(Flatten())
    model_c.add(Dense(100, activation='relu'))
    model_c.add(Dense(NUM_RESULTS, activation='softmax'))
    print(model_c.summary())

    return model_c
# ================================ Commented Out for Now =======================
model_c = create_model()

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='accuracy', patience=1) #Change to monitor='accuracy' if errors
]

model_c.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy']) #Check different ways of calculating loss

# Hyper-parameters
BATCH_SIZE = 400
EPOCHS = 50

# Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
history = model_c.fit(train_X,
                      train_y,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1) #Check that the validation split is not splitting the data in weird ways -> shuffle the data in batches of 50

plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], 'r', label='Accuracy of training data')
plt.plot(history.history['val_accuracy'], 'b', label='Accuracy of validation data')
plt.plot(history.history['loss'], 'r--', label='Loss of training data')
plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
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

#Something appears to be wrong with val_accuracy
'''
