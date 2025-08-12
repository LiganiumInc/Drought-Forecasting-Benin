from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn import preprocessing
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU,InputLayer, Conv1D, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

from deel.puncc.api.prediction import BasePredictor
from deel.puncc.metrics import regression_mean_coverage, regression_sharpness

from keras.models import Sequential
from keras.layers import LSTM, Dense

import numpy as np
#import puncc
from deel.puncc.api.prediction import BasePredictor
from deel.puncc.regression import EnbPI
from deel.puncc.metrics import regression_mean_coverage, regression_sharpness
from deel.puncc.plotting import plot_prediction_intervals

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger

import warnings
warnings.simplefilter('ignore')



def CNN_1D(n_filters, dense_layers, kernel_s, dense_units, input_shape, activations):
    model = Sequential()
    model.add(InputLayer(input_shape))
    model.add(Conv1D(n_filters,
                     kernel_size = kernel_s, padding="same",
                     activation = activations[0]))
    model.add(Flatten())
    for i in range(dense_layers):
      model.add(Dense(units=dense_units[i], activation=activations[i]))
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=[RootMeanSquaredError()])
    return model


def flexible_LSTM(lstm_layers, hidden_units, dense_layers, dense_units, input_shape, activations, if_dropout=False, dropout_val = 0):
    
    model = Sequential()
    model.add(InputLayer(input_shape))
    
    #To stack LSTM layers, we need to change the configuration of the prior LSTM layer to output a 3D array as input for the subsequent layer.
    #We can do this by setting the return_sequences argument on the layer to True (defaults to False). 
    # This will return one output for each input time step and provide a 3D array.
    
    for i in range(lstm_layers):
        model.add(LSTM(hidden_units[i], return_sequences=True))
        
        # return_sequences=True is not necessary for the last LSTM layer
        if i == lstm_layers -1 : 
            model.add(LSTM(hidden_units[i]))
    
    for i in range(dense_layers):
        model.add(Dense(units=dense_units[i], activation=activations[i]))
        
        #  add a Dropout Layer after the first Dense Layer and only if variable dropout is set to True
        if i == 0 and if_dropout == True:
            model.add(Dropout(dropout_val))
        
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=[RootMeanSquaredError()])
    
    return model


def flexible_GRU(gru_layers, hidden_units, dense_layers, dense_units, input_shape, activations, if_dropout=False, dropout_val = 0):
    
    model = Sequential()
    model.add(InputLayer(input_shape))
    
    #To stack gru layers, we need to change the configuration of the prior gru layer to output a 3D array as input for the subsequent layer.
    #We can do this by setting the return_sequences argument on the layer to True (defaults to False). 
    # This will return one output for each input time step and provide a 3D array.
    
    for i in range(gru_layers):
        model.add(GRU(hidden_units[i], return_sequences=True))
        
        # return_sequences=True is not necessary for the last gru layer
        if i == gru_layers -1 : 
            model.add(GRU(hidden_units[i]))
    
    for i in range(dense_layers):
        model.add(Dense(units=dense_units[i], activation=activations[i]))
        
        #  add a Dropout Layer after the first Dense Layer and only if variable dropout is set to True
        if i == 0 and if_dropout == True:
            model.add(Dropout(dropout_val))
        
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=[RootMeanSquaredError()])
    
    return model


def Conv1D_LSTM(conv_filters, conv_kernel_size, lstm_layers, lstm_units, dense_layers, dense_units, input_shape, activations, if_dropout=False, dropout_val=0):
    model = Sequential()
    model.add(InputLayer(input_shape))
    
    # Add Conv1D layer
    model.add(Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, padding="causal", activation=activations[0]))
    
    # Add LSTM layers
    for i in range(lstm_layers):
        if i == lstm_layers - 1:
            model.add(LSTM(lstm_units[i], return_sequences=False))
        else:
            model.add(LSTM(lstm_units[i], return_sequences=True))
    
    # Add dense layers
    for i in range(dense_layers):
        model.add(Dense(units=dense_units[i], activation=activations[i]))  
        if i == 0 and if_dropout:
            model.add(Dropout(dropout_val))
    
    return model


""" CALLBACKS """
def callbacks(train_output_path):

    csv_path = os.path.join(train_output_path)
    callbacks = [
        # ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6,verbose=1),
        CSVLogger(csv_path, append=True),
        # EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    return callbacks

    
def conf_model(model, X_train, y_train, X_val=None, y_val=None, train_output_path=None, epochs=None, batch_size=None, B_ARG=30, mtype="ML"):

    if mtype == "ML":
        rpredictor = BasePredictor(model)

        enbpi = EnbPI(rpredictor, B=B_ARG, agg_func_loo=np.mean, random_state=0)
        enbpi.fit(X_train, y_train)

    else:
        compile_kwargs = {
            "optimizer":"adam", "loss":"mean_squared_error", "metrics": [RootMeanSquaredError()]
        }
        rpredictor = BasePredictor(model, **compile_kwargs)

        enbpi = EnbPI(rpredictor, B=B_ARG, agg_func_loo=np.mean, random_state=0)
        enbpi.fit(X_train, y_train, **{'epochs': epochs, 'batch_size': batch_size,
            'callbacks': callbacks(train_output_path), 'validation_data': (X_val, y_val)
        })
        

    return enbpi
