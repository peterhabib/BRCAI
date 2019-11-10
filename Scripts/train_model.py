#%% Import libraries
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.python.client import device_lib

from tensorflow.keras import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import Normalizer
import pandas
import numpy

import common
import common_scaler
import common_categorical
import common_visualization
import common_pre_post_processing

# print(tf.__version__)
# print(device_lib.list_local_devices())

#%% Load dataset

CSV_PATH = "/home/peter/Desktop/BreastCancerAI/BRCA_AI/TestData.csv"

df = pd.read_csv(CSV_PATH,na_values='?', index_col=False)
df = df.dropna()

print(df.head())
print(df.columns)

X = df[common.X_colum_names]
Y = df[common.Y_colum_names]

print(X.head(), Y.head())
# print(X.values[:,:37])
# breakpoint()

#%% Configure categorical columns
label_encoder, onehot_encoder = common_categorical.create_categorical_feature_encoder(X[common.categorical_column])

#%% Scale data
# Create scaler so that the data is in the same range
x_scaler, y_scaler = common_scaler.create_scaler(X.values[1:,0:4], Y.values)

#%% Split the dataset into different groups
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=50)

print('Training set size: ', len(X_train))
print(X_train.head(), y_train.head())

print('Validation set size: ', len(X_test))
print(X_test.head(), y_test.head())

#%% Transform the data to be used for training the model
# Transform inputs to the format that the model expects
arr_x_train, arr_y_train = common_pre_post_processing.transform_inputs(X_train,
                                                   label_encoder,
                                                   onehot_encoder,
                                                   x_scaler,
                                                   y_train,
                                                   y_scaler)

arr_x_valid, arr_y_valid = common_pre_post_processing.transform_inputs(X_test,
                                                   label_encoder,
                                                   onehot_encoder,
                                                   x_scaler,
                                                   y_test,
                                                   y_scaler)

print('Training shape:', arr_x_train.shape)
print('Training samples: ', arr_x_train.shape[0])
print('Validation samples: ', arr_x_valid.shape[0])

#%% Create the model
def build_model(x_size, y_size):
    model = Sequential()
    model.add(Dense(500, input_shape=(x_size,)))
    # model.add(Dense(200))
    model.add(Dropout(0.2))


    

    # model.add(Dense())
 
    model.add(Activation('sigmoid'))

    model.add(Dense(1))

    model.compile(loss='mean_squared_error',
        optimizer=Adam(0.05),
        metrics=[metrics.mae])

    return(model)

print(arr_x_train.shape[1], arr_y_train.shape[1])

model = build_model(arr_x_train.shape[1], arr_y_train.shape[1])
model.summary()

#%% Configure the training
epochs = 1000
batch_size = 500

print('Epochs: ', epochs)
print('Batch size: ', batch_size)

keras_callbacks = [
    ModelCheckpoint(common.model_checkpoint_file_name,
                    monitor='val_mean_absolute_error',
                    save_best_only=True,
                    verbose=0),
    EarlyStopping(monitor='val_mean_absolute_error', patience=50, verbose=0)
]

#%% Train the model
history = model.fit(arr_x_train, arr_y_train,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=False,
    verbose=2,
    validation_data=(arr_x_valid, arr_y_valid),
    callbacks=keras_callbacks)

model.save(common.model_file_name)

train_score = model.evaluate(arr_x_train, arr_y_train, verbose=2)
valid_score = model.evaluate(arr_x_valid, arr_y_valid, verbose=2)

print('----------|-------------------------------')
print('Train MAE |--> ', round(train_score[1], 4))
print('----------|-------------------------------')
print('Val MAE   |--> ', round(valid_score[1], 4))
print('----------|-------------------------------')
print('Val Loss  |--> ', round(valid_score[0], 4))
print('----------|-------------------------------')
print('Train Loss|--> ', round(train_score[0], 4))
print('----------|-------------------------------')



#%% See training results
common_visualization.plot_history(history.history, x_size=8, y_size=12)



