from pandas import DataFrame
from pandas import concat

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import os
from scipy.stats import pearsonr
#import puncc
from deel.puncc.api.prediction import BasePredictor
from deel.puncc.regression import EnbPI
from deel.puncc.metrics import regression_mean_coverage, regression_sharpness
from deel.puncc.plotting import plot_prediction_intervals

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU,InputLayer, Conv1D, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.simplefilter('ignore')


from termcolor import colored
import pyfiglet
import shutil

sns.set_theme(style="white", context="talk", palette="muted")
# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.size': 14,         
#     'axes.titlesize': 17,
#     'axes.labelsize': 15,    
#     'legend.fontsize': 13,   
#     'xtick.labelsize': 15,
#     'ytick.labelsize': 15,
#     'figure.dpi': 300
# })
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.titlesize': 17,
    'axes.labelsize': 15,
    'legend.fontsize': 13,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'figure.dpi': 300,
    'axes.grid': True,          
    'grid.linestyle': '--',     
    'grid.color': 'gray',       
    'grid.alpha': 0.5
    # 'grid.linewidth': 0.9 for the second case, less clear image, legible image
})

def print_banner(text):
    terminal_width = shutil.get_terminal_size().columns
    print("*" * terminal_width)
    ascii_art = pyfiglet.figlet_format(text, font="big", width=terminal_width)

    centered_ascii_art = "\n".join([
        line.center(terminal_width) for line in ascii_art.splitlines()
    ])

    print(colored(centered_ascii_art, 'magenta')) # cyan, sienna
    print("*" * terminal_width)


def create_lagged_features(data, col_names, n_in=1, n_out=1, dropnan=True):
    
    """
    Source : https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
    
 
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(col_names[j]+'(t-%d)' % (i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [col_names[j]+'(t)'  for j in range(n_vars)]
        else:
            names += [(col_names[j]+'(t+%d)' % (i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def plot_result(trainY, testY, train_predict, test_predict, dates, title):
    actual = np.append(trainY, testY)
    predictions = np.append(train_predict, test_predict)
    rows1 = len(actual)
    rows2 = len(predictions)
    plt.figure(figsize=(18, 6), dpi=300)
    plt.plot(range(rows1), actual)
    plt.plot(range(rows2), predictions)
    plt.title(title)
    plt.axvline(x=len(trainY), color='r')
    dates_ticks = [dates[i] for i in range(0,rows1,25) ]
    plt.xticks(ticks=range(0,rows1,25),labels=dates_ticks)
    plt.legend(['Actual', 'Predictions'])
    plt.xlabel('Dates')
    plt.ylabel('SPI6')

    
def plot_results_test(y_test, y_pred, title):
    
    dates = list(y_test.index)
    nb_elems = len(dates)
    plt.figure(figsize=(15, 6), dpi=300)
    plt.plot(y_test.values)
    plt.plot(y_pred)
    plt.title(title)
    dates_ticks = [dates[i] for i in range(0,nb_elems,10) ]
    plt.xticks(ticks=range(0,nb_elems,10),labels=dates_ticks)
    plt.legend(['Actual', 'Predictions'])
    plt.xlabel('Dates')
    plt.ylabel('SPI6')
    
    
# Plot the validation and training curves separately
def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.
  """
  loss = history.history["loss"]
  val_loss = history.history["val_loss"]

  rmse = history.history["root_mean_squared_error"]
  val_rmse = history.history["val_root_mean_squared_error"]

  epochs = range(len(history.history["loss"])) # how many epochs did we run for?

  # Plot loss
  plt.figure(figsize=(10, 3))
  plt.plot(epochs, loss, label="training_loss")
  plt.plot(epochs, val_loss, label="val_loss")
  plt.title("loss")
  plt.xlabel("epochs")
  plt.legend()

  # Plot accuracy
  plt.figure(figsize=(10, 3))
  plt.plot(epochs, rmse, label="training_rmse")
  plt.plot(epochs, val_rmse, label="val_rmse")
  plt.title("RMSE")
  plt.xlabel("epochs")
  plt.legend();
    
def dl_plot_results_test(y_test,y_pred,dat):
    
    dates = dat
    nb_elems = len(dates)
    plt.figure(figsize=(15, 6), dpi=300)
    plt.plot(y_test)
    plt.plot(y_pred)
    dates_ticks = [dates[i] for i in range(0,nb_elems,10) ]
    plt.xticks(ticks=range(0,nb_elems,10),labels=dates_ticks)
    plt.legend(['Actual', 'Predictions'])
    plt.xlabel('Dates')
    plt.ylabel('SPI6')


# compute rsquared
def r_squared(predicted, actual):
    """
    Compute R² (coefficient of determination) for predicted and actual values
    :param predicted: array of predicted values
    :param actual: array of actual values
    :return: R² value
    """
    mean_actual = np.mean(actual)
    sse = np.sum((actual - predicted) ** 2)
    sst = np.sum((actual - mean_actual) ** 2)
    r2 = 1 - (sse / sst)
    return r2

# RMSE 
def RMSE(predicted, actual):
    """
    Compute RMSE (root mean squared error) for predicted and actual values
    :param predicted: array of predicted values
    :param actual: array of actual values
    :return: RMSE value
    """
    return np.sqrt(np.mean((predicted - actual) ** 2))

def evaluate_preds(xtest, y_true, y_pred, y_pred_lower, y_pred_upper):
    
    # Calculate various metrics
    # Rsquared 
    rsq = r_squared(y_pred,y_true)
    

    # RMSE
    rmse = RMSE(y_pred,y_true)
    
    # MAE
    mae = np.mean(np.abs(y_pred - y_true))

    # coverage
    coverage = regression_mean_coverage(y_true, y_pred_lower, y_pred_upper)

    # sharpness
    width = regression_sharpness(y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper)
  
    return {"R2": rsq,
          "RMSE": rmse,
          "MAE": mae,
          "Coverage": coverage,
          "Width": width
         }
    
def classement(results, metric):
    
    asc = True if metric == 'RMSE' or metric == 'MAE' or metric == 'Width' or metric == 'CO2_EMISSIONS' else False
    serie = results[metric]
    ranking = serie.sort_values(ascending=asc)
    return list(ranking.index)


def borda_count_ranking(results):
    
    models = list(results.index)
    rankings = {model: 0 for model in models}  # Créer un dictionnaire pour stocker les points de chaque modèle
    
    # Préférences des électeurs par ordre de meilleurs performances (RMSE)
    rmse_preferences = classement(results, 'RMSE')
    mae_preferences = classement(results, 'MAE')
    r2_preferences = classement(results, 'R2')
    coverage_preferences = classement(results, "Coverage")
    width_preferences = classement(results, "Width")
    CO2_EMISSIONS_preferences = classement(results, 'CO2_EMISSIONS')
    
    # Attribution des points en fonction des préférences des électeurs
    num_models = len(models)
    for model in models:
        rmse_points = np.abs(rmse_preferences.index(model) - len(models))
        mae_points = np.abs(mae_preferences.index(model) - len(models))
        r2_points = np.abs(r2_preferences.index(model) - len(models))
        coverage_points = np.abs(coverage_preferences.index(model) - len(models))
        width_points = np.abs(width_preferences.index(model) - len(models))
        CO2_EMISSIONS_points = np.abs(CO2_EMISSIONS_preferences.index(model) - len(models))
        
        rankings[model] += (rmse_points + mae_points + r2_points + coverage_points + width_points + CO2_EMISSIONS_points)
    
    # Création du DataFrame avec les noms des modèles et les points
    df = pd.DataFrame({'Models': list(rankings.keys()), 'Points': list(rankings.values())})
    
    # Trier le DataFrame par ordre décroissant des points
    df = df.sort_values(by='Points', ascending=False).reset_index(drop=True)
    
    return df


def plot_predicted_interval(y_test, y_pred, y_pred_lower, y_pred_upper, title):

    dates = list(y_test.index)
    nb_elems = len(dates)
    dates_ticks = [dates[i] for i in range(0,nb_elems,10) ]

    ax = plot_prediction_intervals(
    y_true=y_test,
    y_pred=y_pred,
    y_pred_lower=y_pred_lower,
    y_pred_upper=y_pred_upper,
    figsize=(20, 10),
    loc="best")
    ax.set_xticks(ticks=range(0,nb_elems,10),labels=dates_ticks)
    ax.set_title(title)
    ax.set_xlabel('Dates')
    ax.set_ylabel('SPI6')

    # Improve the plots, xticks bald, and size


def set_seed(seed):
    # Set the seed for Python's built-in random module
    random.seed(seed)
    
    # Set the seed for NumPy
    np.random.seed(seed)
    
    # Set the seed for TensorFlow
    tf.random.set_seed(seed)
    
    # For TensorFlow 1.x, you might need to set the session-level seed
    # Uncomment the following lines if you are using TensorFlow 1.x
    # from tensorflow.compat.v1.keras import backend as K
    # session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    # tf.compat.v1.keras.backend.set_session(sess)
    
    # Ensure reproducibility with TensorFlow by controlling parallel threads
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set the seed for scikit-learn
    # Note: scikit-learn relies on NumPy's random state, so setting the NumPy seed is sufficient for reproducibility in scikit-learn
    # However, if you use other libraries that depend on NumPy's random state, you should ensure their seed is also s


def perform_ranking(results, metric):
    
    asc = True if metric == 'RMSE' or metric == 'MAE' or metric == 'Width' else False
    serie = results[metric]
    ranking = serie.sort_values(ascending=asc)
    return list(ranking.index)


def borda_count(results):
    
    models = list(results.index)
    rankings = {model: 0 for model in models}  # Créer un dictionnaire pour stocker les points de chaque modèle
    
    # Préférences des électeurs par ordre de meilleurs performances (RMSE)
    rmse_preferences = perform_ranking(results, 'RMSE')
    mae_preferences = perform_ranking(results, 'MAE')
    r2_preferences = perform_ranking(results, 'R2')
    coverage_preferences = perform_ranking(results, "Coverage")
    width_preferences = perform_ranking(results, "Width")
    # CO2_EMISSIONS_preferences = classement(results, 'CO2_EMISSIONS')
    
    # Attribution des points en fonction des préférences des électeurs
    num_models = len(models)
    for model in models:
        rmse_points = np.abs(rmse_preferences.index(model) - len(models))
        mae_points = np.abs(mae_preferences.index(model) - len(models))
        r2_points = np.abs(r2_preferences.index(model) - len(models))
        coverage_points = np.abs(coverage_preferences.index(model) - len(models))
        width_points = np.abs(width_preferences.index(model) - len(models))
        # CO2_EMISSIONS_points = np.abs(CO2_EMISSIONS_preferences.index(model) - len(models))
        
        rankings[model] += (rmse_points + mae_points + r2_points + coverage_points + width_points)
    
    # Création du DataFrame avec les noms des modèles et les points
    df = pd.DataFrame({'Models': list(rankings.keys()), 'Points': list(rankings.values())})
    
    # Trier le DataFrame par ordre décroissant des points
    df = df.sort_values(by='Points', ascending=False).reset_index(drop=True)
    
    return df
