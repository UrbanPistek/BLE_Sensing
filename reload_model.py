from tensorflow import keras
import pandas as pd
from tqdm import tqdm
from scipy import stats
import numpy as np
import random
import os
from keras.utils import np_utils
from matplotlib import pyplot as plt

TIME_STEPS = 200 #change based on training
STEP_DISTANCE = 10 #change based on training
NUM_PARAMETERS = 4
N_FEATURES = 4
INPUT_SHAPE = NUM_PARAMETERS*TIME_STEPS
NUM_RESULTS = 3
DATASET = "dataset2"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Just disables the warning, doesn't enable AVX/FMA

def load_dataset():
    data = pd.read_csv('dataset2_test.csv') #optimize, best dataset = dataset2
    # Normalize the Data - Note Data is all negative so dividing by istelf twice will make it positive
    data['B1'] = data['B1'] / data['B1'].min()
    data['B2'] = data['B2'] / data['B2'].min()
    data['B3'] = data['B3'] / data['B3'].min()
    data['B4'] = data['B4'] / data['B4'].min()

    data['Label'].replace({"Empty": 0, "One person": 1, "Two people": 2}, inplace=True) #Replace Strings with values

    # Round numbers
    data = data.round({'B1': 6, 'B2': 6, 'B3': 6, 'B4': 6})

    return data

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

def prep_data(X, y):
    train_X = X[0:len(X)]
    train_y = y[0:len(y)]
    train_X = train_X.reshape(train_X.shape[0], INPUT_SHAPE)
    train_X = train_X.astype('float32')
    train_y= train_y.astype('float32')
    train_y = np_utils.to_categorical(train_y, NUM_RESULTS)

    return train_X, train_y

def shuffle_data(X, y):
    c = list(zip(X, y))
    random.shuffle(c)
    X, y = zip(*c)
    X = np.asarray(X)
    y = np.asarray(y)

    return X, y

def run_steps():
    scores = []
    model = keras.models.load_model('model_epochs175_batch25_time200_step20dataset2_ker100_100_drop_val0.25.h5')
    model.summary()
    max_value = 0

    for idx in tqdm(range(100)): #change to match TIME_STEPS
        data = load_dataset()
        X, y = make_training_data(data, TIME_STEPS, idx+1)
        train_X, train_y = shuffle_data(X, y)
        train_X, train_y = prep_data(train_X, train_y)
        score = model.evaluate(train_X, train_y, verbose = 0)
        scores.append(score[1])

        if max_value < scores[idx]:
            max_value = scores[idx]

    plt.plot(scores)
    plt.title('Out of Sample accuracy versus Step Distance')
    plt.ylabel('Accuracy')
    plt.xlabel('Step Distance')
    plt.show()
    print("Max Value: ", max_value)

    return scores

def get_score():
    data = load_dataset()
    X, y = make_training_data(data, TIME_STEPS, STEP_DISTANCE)
    train_X, train_y = shuffle_data(X, y)
    train_X, train_y = prep_data(train_X, train_y)

    model = keras.models.load_model('model_epochs500_batch25_time200_step20dataset2_ker100_100_drop_val0.25.h5')
    model.summary()
    score = model.evaluate(train_X, train_y, verbose = 0)
    print(score)
    print("Score: ", "%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

#get_score()
acc = run_steps()
print(acc)
acc_df = pd.DataFrame(acc)
#print(acc_df)
#print("Average: ", acc_df['0'].mean()) #get mean value