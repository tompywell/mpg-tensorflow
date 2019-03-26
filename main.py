from __future__ import absolute_import, division
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# define column names
column_names = ['mpg','cylinders','displacement','horsepower','weight', 'acceleration',
                'model year', 'origin', 'name']

# read in the data from csv file
data = pd.read_csv('./data/auto-mpg.csv', names=column_names, na_values = '?', sep=',')

# get rid of the name column, it has no value
data.pop('name')

# get rid of any rows with any na values
data = data.dropna()

def one_hot_origin(data):
    origin = data.pop('origin')
    data['usa'] = (origin == 1) * 1.0
    data['eur'] = (origin == 2) * 1.0
    data['jap'] = (origin == 3) * 1.0
    return data

# replace origin column with 3 one-hot encodings
data = one_hot_origin(data);
# origin = data.pop('origin')
# data['usa'] = (origin == 1) * 1.0
# data['eur'] = (origin == 2) * 1.0
# data['jap'] = (origin == 3) * 1.0

# split the data into traininging data and testinging data
training_data = data.sample(frac=0.9, random_state=0)
testing_data = data.drop(training_data.index)

# get common statistics from the data
# mean and standard deviation will be used to normalize the data
data_stats = training_data.describe()

# remove any info about label/mpg
data_stats.pop("mpg")

# transpose the stats, so features are now rows and statistical measures are now columns
data_stats = data_stats.transpose()

# get the training labels from the training data
training_labels = training_data.pop('mpg')

# get the testing labels from the testing data
testing_labels = testing_data.pop('mpg')

# a function used to normalize all numbers
# meaning their values will now be their deviation from the mean
def normalize(data):
  return (data - data_stats['mean']) / data_stats['std']

# normalize the training data
training_data = normalize(training_data)

# normalize the testing data
testing_data = normalize(testing_data)

# function to build the model
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(training_data.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error', optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

# build the model
model = build_model()

# define how many epochs to use
NUM_EPOCHS = 500

# fit the model to the training data and training labels
model.fit(training_data, training_labels, epochs=NUM_EPOCHS, validation_split=0.1, verbose=0)

# evaluate the model on the testing set, finding loss, mean absolute error, and mean squared error
(loss, mae, mse) = model.evaluate(testing_data, testing_labels, verbose=0)

# print an evaluation metric
print("\nmean squared error: " + str(mse) + " mpg")
print("mean absolute error: " + str(mae) + " mpg")

# going to predict some labels based off some made-up car features
predict = pd.read_csv('./data/predict.csv', names=column_names, na_values = '?', sep=',')

# get rid of name feature, its useless
predict.pop('name')

# encode origin as 3 one-hot features
predict = one_hot_origin(predict)

# normalize the features using the stats from the training data
normalized_predict = normalize(predict)

# get rid of mpg, this is our label to predict
normalized_predict.pop('mpg')

# predict
prediction = model.predict(normalized_predict)

predict['mpg'] = prediction

print('\npredictions are as follows:\n')

#print prediction labels and features
print(predict)
