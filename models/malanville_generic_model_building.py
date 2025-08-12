import pickle, os, glob, ast
import os
from utils import *
from models import * 
import yaml
from codecarbon import EmissionsTracker
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import plot_model

seed = 42
set_seed(seed)

with open('../configs/config_malanville.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

from taylor_diagram import plot_taylor_diagram, print_statistics_summary 

models_predictions = {} # Dictionary to hold model predictions 


print('Config file: {}'.format(config))

"""Extract data params"""
data_params = config['data_params']

B_ARG = data_params['B_ARG']

"""Create a directory to save results"""
save_path = data_params['save_path']

"""Create subdirectories to save results"""
dirs = [data_params['city']+"/Images/",
        data_params['city']+"/Models/",
        data_params['city']+"/Excel/"]

for path in dirs:
    if not os.path.exists(save_path+path):
        os.makedirs(save_path+path)

"""Lagged data path"""
lagged_data_path = data_params['data_path'] + 'lagged/' + data_params['city'] + '_lagged_raw.csv'

target_column = data_params['target_column']

dataset = pd.read_csv(lagged_data_path)

dataset.set_index('DATE', inplace=True)

print(dataset)

"""Convert split_date to datetime"""
split_date = data_params['split_date']

target = dataset[target_column]
dataset.drop(target_column, axis=1,inplace=True)

"""Split the DataFrame"""
X_trains, X_tests  = dataset.loc[:split_date], dataset.loc[split_date:]
y_train, y_test = target.loc[:split_date], target.loc[split_date:]
columns = X_trains.columns

"""Normalization"""
feature_range = (-1, 1)
scaler = MinMaxScaler(feature_range=feature_range).fit(X_trains)

X_trains = pd.DataFrame(data=scaler.transform(X_trains), columns=columns)
X_tests = pd.DataFrame(data=scaler.transform(X_tests), columns=columns)

X_train, X_test = np.array(X_trains), np.array(X_tests)

times = list(dataset.index)


print_banner('Data shape')
print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")


print_banner("Training Linear Regression")

tracker = EmissionsTracker(output_dir=os.path.join(os.getcwd(), save_path, data_params['city']),
    output_file='CO2_detailed_output.csv',
    project_name='Linear Regression'
)
tracker.start()
linear_reg_model = conf_model( LinearRegression(), X_train, y_train, mtype='ML', B_ARG=B_ARG)
emissions = tracker.stop()

y_pred1, y_pred_lower, y_pred_upper = linear_reg_model.predict(X_test, alpha=.1, y_true=y_test, s=None)
y_train_pred1, y_train_pred_lower, y_train_pred_upper = linear_reg_model.predict(X_train, alpha=.1, y_true=y_train, s=None)

plot_result(y_train, y_test, y_train_pred1, y_pred1,dates = list(dataset.index), title='Linear Regression')
plt.savefig(save_path + dirs[0] + 'lr_model_with_lags.png')

plot_results_test(y_test, y_pred1, title='Linear Regression')
plt.savefig(save_path + dirs[0] + 'lr_model_with_lags_test_part.png')

plot_predicted_interval(y_test, y_pred1, y_pred_lower, y_pred_upper, "Linear Regression")
plt.savefig(save_path + dirs[0] + 'lr_model_with_lags_test_part_cf.png')

linear_lr_results = evaluate_preds(y_test, y_pred1, y_pred_lower, y_pred_upper)

linear_lr_results['CO2_EMISSIONS'] = emissions

models_predictions['Linear Regression'] = y_pred1
print("Linear regression model predictions shape: ", y_pred1.shape)
print("y_test shape: ", y_test.shape)

print("Linear regression results: ", linear_lr_results)





print_banner("Training Ridge")

tracker = EmissionsTracker(output_dir=os.path.join(os.getcwd(), save_path, data_params['city']),
    output_file='CO2_detailed_output.csv',
    project_name='Ridge'
)
tracker.start()
ridge_model = conf_model(Ridge(), X_train, y_train, mtype = "ML", B_ARG=B_ARG)
emissions = tracker.stop()

y_pred2, y_pred_lower, y_pred_upper = ridge_model.predict(X_test, alpha=.1, y_true=y_test, s=None)
y_train_pred2, y_train_pred_lower, y_train_pred_upper = ridge_model.predict(X_train, alpha=.1, y_true=y_train, s=None)

plot_result(y_train, y_test, y_train_pred2, y_pred2,dates = list(dataset.index), title='Ridge')
plt.savefig(save_path + dirs[0] + 'ridge_model_with_lags.png')

plot_results_test(y_test, y_pred2, title='Ridge')
plt.savefig(save_path + dirs[0] + 'ridge_model_with_lags_test_part.png')

plot_predicted_interval(y_test, y_pred2, y_pred_lower, y_pred_upper, "Ridge")
plt.savefig(save_path + dirs[0] + 'ridge_model_with_lags_test_part_cf.png')

ridge_results = evaluate_preds(y_test, y_pred2, y_pred_lower, y_pred_upper)

ridge_results['CO2_EMISSIONS'] = emissions

models_predictions['Ridge'] = y_pred2

print("Ridge results: ", ridge_results)





print_banner("Training RandomForest")

tracker = EmissionsTracker(output_dir=os.path.join(os.getcwd(), save_path, data_params['city']),
    output_file='CO2_detailed_output.csv',
    project_name='RandomForest'
)
tracker.start()
rforest_model = conf_model(RandomForestRegressor(max_depth=3, random_state=0), X_train, y_train, mtype = "ML", B_ARG=B_ARG)
emissions = tracker.stop()

y_pred3, y_pred_lower, y_pred_upper = rforest_model.predict(X_test, alpha=.1, y_true=y_test, s=None)
y_train_pred3, y_train_pred_lower, y_train_pred_upper = rforest_model.predict(X_train, alpha=.1, y_true=y_train, s=None)

plot_result(y_train, y_test, y_train_pred3, y_pred3,dates = list(dataset.index), title='Random Forest')
plt.savefig(save_path + dirs[0] + 'random_forest_model_with_lags.png')

plot_results_test(y_test, y_pred3, title='Random Forest')
plt.savefig(save_path + dirs[0] + 'random_forest_model_with_lags_test_part.png')

plot_predicted_interval(y_test, y_pred3, y_pred_lower, y_pred_upper, "Random Forest")
plt.savefig(save_path + dirs[0] + 'random_forest_model_with_lags_test_part_cf.png')

rforest_results = evaluate_preds(y_test, y_pred3, y_pred_lower, y_pred_upper)

rforest_results['CO2_EMISSIONS'] = emissions

models_predictions['Random Forest'] = y_pred3

print("Random forest regressor results: ", rforest_results)


print_banner("Training XGBoost")

tracker = EmissionsTracker(output_dir=os.path.join(os.getcwd(), save_path, data_params['city']),
    output_file='CO2_detailed_output.csv',
    project_name='Xgboost'
)
tracker.start()
xgboost_model = conf_model(XGBRegressor(), X_train, y_train, mtype = "ML", B_ARG=B_ARG)
emissions = tracker.stop()

y_pred4, y_pred_lower, y_pred_upper = xgboost_model.predict(X_test, alpha=.1, y_true=y_test, s=None)

y_train_pred4, y_train_pred_lower, y_train_pred_upper = xgboost_model.predict(X_train, alpha=.1, y_true=y_train, s=None)

plot_result(y_train, y_test, y_train_pred4, y_pred4,dates = list(dataset.index), title='Xgboost')
plt.savefig(save_path + dirs[0] + 'xgb_model_with_lags.png')

plot_results_test(y_test, y_pred4, title='Xgboost')
plt.savefig(save_path + dirs[0] + 'xgb_model_with_lags_test_part.png')

plot_predicted_interval(y_test, y_pred4, y_pred_lower, y_pred_upper, "Xgboost")
plt.savefig(save_path + dirs[0] + 'xgb_model_with_lags_test_part_cf.png')

xgboost_results = evaluate_preds(y_test, y_pred4, y_pred_lower, y_pred_upper)

xgboost_results['CO2_EMISSIONS'] = emissions

models_predictions['XGBoost'] = y_pred4

print("Xgboost results: ", xgboost_results)


print_banner("Training LightGBM")

tracker = EmissionsTracker(output_dir=os.path.join(os.getcwd(), save_path, data_params['city']),
    output_file='CO2_detailed_output.csv',
    project_name='Lightgbm'
)
tracker.start()
lightgbm_model = conf_model(LGBMRegressor(verbose=-1), X_train, y_train, mtype = "ML", B_ARG=B_ARG)
emissions = tracker.stop()

y_pred5, y_pred_lower, y_pred_upper = lightgbm_model.predict(X_test, alpha=.1, y_true=y_test, s=None)
y_train_pred5, y_train_pred_lower, y_train_pred_upper = lightgbm_model.predict(X_train, alpha=.1, y_true=y_train, s=None)

plot_result(y_train, y_test, y_train_pred5, y_pred5,dates = list(dataset.index), title='Lightgbm')
plt.savefig(save_path + dirs[0] + 'lgbm_model_with_lags.png')

plot_results_test(y_test, y_pred5, title='Lightgbm')
plt.savefig(save_path + dirs[0] + 'lgbm_model_with_lags_test_part.png')

plot_predicted_interval(y_test, y_pred5, y_pred_lower, y_pred_upper, "Lightgbm")
plt.savefig(save_path + dirs[0] + 'lgbm_model_with_lags_test_part_cf.png')

lightgbm_results = evaluate_preds(y_test, y_pred5, y_pred_lower, y_pred_upper)

lightgbm_results['CO2_EMISSIONS'] = emissions

models_predictions['LightGBM'] = y_pred5

print("Lightgbm results: ", lightgbm_results)


print_banner("Training SVR")

tracker = EmissionsTracker(
    output_dir=os.path.join(os.getcwd(), save_path, data_params['city']),
    output_file='CO2_detailed_output.csv',
    project_name='SVR'
)
tracker.start()
svr_model = conf_model(SVR(), X_train, y_train, mtype = "ML", B_ARG=B_ARG)
emissions = tracker.stop()

y_pred6, y_pred_lower, y_pred_upper = svr_model.predict(X_tests, alpha=.1, y_true=y_test, s=None)

y_train_pred6, y_train_pred_lower, y_train_pred_upper = svr_model.predict(X_train, alpha=.1, y_true=y_train, s=None)

plot_result(y_train, y_test, y_train_pred6, y_pred6,dates = list(dataset.index), title='SVR')
plt.savefig(save_path + dirs[0] + 'svr_model_with_lags.png')

plot_results_test(y_test, y_pred6, title='SVR')
plt.savefig(save_path + dirs[0] + 'svr_model_with_lags_test_part.png')

plot_predicted_interval(y_test, y_pred6, y_pred_lower, y_pred_upper, "SVR")
plt.savefig(save_path + dirs[0] + 'SVR_model_with_lags_test_part_cf.png')

svr_results = evaluate_preds(y_test, y_pred6, y_pred_lower, y_pred_upper)

svr_results['CO2_EMISSIONS'] = emissions

models_predictions['SVR'] = y_pred6

print("SVR results: ", svr_results)



# ---------------------------------- Deep Learning ------------------------------------------
# --- Step 1: Load raw data ---
dataset_path = data_params['data_path'] + 'raw/' + data_params['dataset']
dataset = pd.read_csv(dataset_path)
dataset['DATE'] = pd.to_datetime(dataset['DATE'], dayfirst=True)
dataset = dataset.set_index('DATE')

# remove the first 5 rows that are Nan in the target column because they are used to compute the SPI6
dataset = dataset.iloc[5:]

# --- Step 2: Select columns ---
target_column = 'SPI6'
selected_columns = ['PS', 'T2M', 'RH2M', 'WS2M', 'GWETPROF', 'PRECTOTCORR_SUM', 'SPI6']

# --- Step 3: Normalize features and target separately ---
split_date = pd.to_datetime(data_params['split_date'])
# dataset_train = dataset.loc[:split_date]
# dataset_test = dataset.loc[split_date:]
dataset_train = dataset.loc[dataset.index < split_date]
dataset_test = dataset.loc[dataset.index >= split_date]

features_only = [col for col in selected_columns if col != target_column]

scaler_X = MinMaxScaler(feature_range=(-1, 1)).fit(dataset_train[features_only])
scaler_y = MinMaxScaler(feature_range=(-1, 1)).fit(dataset_train[[target_column]])

dataset_train_scaled = dataset_train.copy()
dataset_test_scaled = dataset_test.copy()

dataset_train_scaled[features_only] = scaler_X.transform(dataset_train[features_only])
dataset_test_scaled[features_only] = scaler_X.transform(dataset_test[features_only])
dataset_train_scaled[[target_column]] = scaler_y.transform(dataset_train[[target_column]])
dataset_test_scaled[[target_column]] = scaler_y.transform(dataset_test[[target_column]])

dataset_scaled = pd.concat([dataset_train_scaled, dataset_test_scaled])

dataset_scaled = dataset_scaled[5:]

# --- Step 4: Create sequences ---
window_size = data_params.get("window_size", 5)
X_all, y_all = create_dl_sequences(dataset_scaled, target_column, selected_columns, window_size)

# --- Step 5: Split train/test ---
split_idx = dataset_scaled.index.to_list().index(split_date) - window_size
X_train, X_test = X_all[:split_idx], X_all[split_idx:]
y_train, y_test = y_all[:split_idx], y_all[split_idx:]

# --- Step 6: Validation split ---
val_split_idx = int(0.8 * len(X_train))
X_val, y_val = X_train[val_split_idx:], y_train[val_split_idx:]
X_train, y_train = X_train[:val_split_idx], y_train[:val_split_idx]

# --- Step 7: Training ---
print_banner('DL dataset shape')
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

print_banner("Training Conv1D model")

conv1d_model_params = config['conv1d_model']
file_name = "conv1D_train_output.csv"
train_output_path = os.path.join(save_path, data_params['city'], file_name)

CNN = CNN_1D(
    n_filters=conv1d_model_params['cnn_units'],
    dense_layers=conv1d_model_params['dense_layers'],
    kernel_s=conv1d_model_params['kernel_size'],
    dense_units=conv1d_model_params['dense_units'],
    input_shape=X_train.shape[1:],
    activations=conv1d_model_params['activ']
)

print(CNN.summary())
# plot_model(CNN, to_file='conv1D_train_output.png', show_shapes=True, show_layer_names=True)

tracker = EmissionsTracker(output_dir=os.path.join(os.getcwd(), save_path, data_params['city']),
    output_file='CO2_detailed_output.csv', project_name='Conv1D-model')
tracker.start()
CNNP = conf_model(CNN, X_train, y_train, X_val, y_val, train_output_path, mtype="DL", 
    epochs=conv1d_model_params['epochs'], batch_size=conv1d_model_params['batch_size'], B_ARG=B_ARG
)
emissions = tracker.stop()

y_test = y_all[split_idx:].copy()   
Y_pred, y_pred_lower, y_pred_upper = CNNP.predict(X_test, alpha=.1, y_true=y_test , s=None)

# # Inverse transform
Y_pred = scaler_y.inverse_transform(Y_pred.reshape(-1, 1))
y_pred_lower = scaler_y.inverse_transform(y_pred_lower.reshape(-1, 1))
y_pred_upper = scaler_y.inverse_transform(y_pred_upper.reshape(-1, 1))
y_test_inv  = scaler_y.inverse_transform(y_test.reshape(-1, 1))  

conv1D_results = evaluate_preds(y_test_inv  , Y_pred, y_pred_lower, y_pred_upper)

plot_predicted_interval_dl(y_test_inv  , Y_pred, y_pred_lower, y_pred_upper,dates = list(dataset_test.index),title= "CNN")

plt.savefig(save_path + dirs[0] + 'cnn_model_with_lags_test_part_cf.png')

y_train = y_train[:val_split_idx]
Y_train_pred, y_train_pred_lower, y_train_pred_upper = CNNP.predict(X_train, alpha=.1, y_true=y_train, s=None)
# # Inverse transform
Y_train_pred = scaler_y.inverse_transform(Y_train_pred.reshape(-1, 1))
y_train_pred_lower = scaler_y.inverse_transform(y_train_pred_lower.reshape(-1, 1))
y_train_pred_upper = scaler_y.inverse_transform(y_train_pred_upper.reshape(-1, 1))
y_train_inv = scaler_y.inverse_transform(y_train.reshape(-1, 1))
plot_result(y_train_inv, y_test_inv  , Y_train_pred, Y_pred, dates = list(dataset.index), title='CNN')

conv1D_results['CO2_EMISSIONS'] = emissions

models_predictions['Conv1D'] = Y_pred 

print(Y_pred.shape)
print(y_test_inv.shape)

print("conv1D_results: ", conv1D_results)

# --- Updated LSTM Model Block ---

print_banner("Training LSTM")

lstm_model_params = config['lstm_model']
file_name = "LSTM_train_output.csv"
train_output_path = os.path.join(save_path, data_params['city'], file_name)

lstm_model_instance = flexible_LSTM(
    lstm_layers=lstm_model_params['lstm_layers'],
    hidden_units=lstm_model_params['lstm_units'],
    dense_layers=lstm_model_params['dense_layers'],
    dense_units=lstm_model_params['dense_units'],
    input_shape=X_train.shape[1:],
    activations=lstm_model_params['activ'],
    if_dropout=lstm_model_params['dropout'],
    dropout_val=lstm_model_params['dropout_val']
)

print(lstm_model_instance.summary())

tracker = EmissionsTracker(output_dir=os.path.join(os.getcwd(), save_path, data_params['city']),
    output_file='CO2_detailed_output.csv', project_name='LSTM-model')
tracker.start()

LSTM = conf_model(lstm_model_instance, X_train, y_train, X_val, y_val, train_output_path, mtype="DL", 
    epochs=lstm_model_params['epochs'], batch_size=lstm_model_params['batch_size'], B_ARG=B_ARG)

emissions = tracker.stop()

y_test = y_all[split_idx:].copy()   
Y_pred, y_pred_lower, y_pred_upper = LSTM.predict(X_test, alpha=.1, y_true=y_test, s=None)

Y_pred = scaler_y.inverse_transform(Y_pred.reshape(-1, 1))
y_pred_lower = scaler_y.inverse_transform(y_pred_lower.reshape(-1, 1))
y_pred_upper = scaler_y.inverse_transform(y_pred_upper.reshape(-1, 1))
y_test_inv   = scaler_y.inverse_transform(y_test.reshape(-1, 1))

lstm_results = evaluate_preds(y_test_inv  , Y_pred, y_pred_lower, y_pred_upper)
plot_predicted_interval_dl(y_test_inv  , Y_pred, y_pred_lower, y_pred_upper, dates=list(dataset_test.index), title="LSTM")
plt.savefig(save_path + dirs[0] + 'lstm_model_with_lags_test_part_cf.png')

y_train = y_train[:val_split_idx]
Y_train_pred, y_train_pred_lower, y_train_pred_upper = LSTM.predict(X_train, alpha=.1, y_true=y_train, s=None)
Y_train_pred = scaler_y.inverse_transform(Y_train_pred.reshape(-1, 1))
y_train_pred_lower = scaler_y.inverse_transform(y_train_pred_lower.reshape(-1, 1))
y_train_pred_upper = scaler_y.inverse_transform(y_train_pred_upper.reshape(-1, 1))
y_train_inv = scaler_y.inverse_transform(y_train.reshape(-1, 1))

plot_result(y_train_inv, y_test_inv  , Y_train_pred, Y_pred, dates=list(dataset.index), title='LSTM')

lstm_results['CO2_EMISSIONS'] = emissions

models_predictions['LSTM'] = Y_pred

print("LSTM results: ", lstm_results)


### --- Updated GRU Model Block ---

print_banner("Training GRU")

gru_model_params = config['gru_model']
file_name = "GRU_train_output.csv"
train_output_path = os.path.join(save_path, data_params['city'], file_name)

gru_model = flexible_GRU(
    gru_layers=gru_model_params['gru_layers'],
    hidden_units=gru_model_params['gru_units'],
    dense_layers=gru_model_params['dense_layers'],
    dense_units=gru_model_params['dense_units'],
    input_shape=X_train.shape[1:],
    activations=gru_model_params['activ'],
    if_dropout=gru_model_params['dropout'],
    dropout_val=gru_model_params['dropout_val']
)

print(gru_model.summary())

tracker = EmissionsTracker(output_dir=os.path.join(os.getcwd(), save_path, data_params['city']),
    output_file='CO2_detailed_output.csv', project_name='GRU-model')
tracker.start()

GRU = conf_model(gru_model, X_train, y_train, X_val, y_val, train_output_path, mtype="DL",
    epochs=gru_model_params['epochs'], batch_size=gru_model_params['batch_size'], B_ARG=B_ARG)

emissions = tracker.stop()

y_test = y_all[split_idx:].copy()
Y_pred, y_pred_lower, y_pred_upper = GRU.predict(X_test, alpha=.1, y_true=y_test, s=None)
Y_pred = scaler_y.inverse_transform(Y_pred.reshape(-1, 1))
y_pred_lower = scaler_y.inverse_transform(y_pred_lower.reshape(-1, 1))
y_pred_upper = scaler_y.inverse_transform(y_pred_upper.reshape(-1, 1))
y_test_inv   = scaler_y.inverse_transform(y_test.reshape(-1, 1))

gru_results = evaluate_preds(y_test_inv  , Y_pred, y_pred_lower, y_pred_upper)
plot_predicted_interval_dl(y_test_inv  , Y_pred, y_pred_lower, y_pred_upper, dates=list(dataset_test.index), title="GRU")
plt.savefig(save_path + dirs[0] + 'gru_model_with_lags_test_part_cf.png')

y_train = y_train[:val_split_idx]
Y_train_pred, y_train_pred_lower, y_train_pred_upper = GRU.predict(X_train, alpha=.1, y_true=y_train, s=None)
Y_train_pred = scaler_y.inverse_transform(Y_train_pred.reshape(-1, 1))
y_train_pred_lower = scaler_y.inverse_transform(y_train_pred_lower.reshape(-1, 1))
y_train_pred_upper = scaler_y.inverse_transform(y_train_pred_upper.reshape(-1, 1))
y_train_inv = scaler_y.inverse_transform(y_train.reshape(-1, 1))

plot_result(y_train_inv, y_test_inv  , Y_train_pred, Y_pred, dates=list(dataset.index), title='GRU')
emissions = tracker.stop()
gru_results['CO2_EMISSIONS'] = emissions

models_predictions['GRU'] = Y_pred

print("GRU results: ", gru_results)


# ### --- Updated Conv1D-LSTM Model Block ---

print_banner("Training Conv1D-LSTM")

conv1d_lstm_model_params = config['conv1d_lstm_model']
file_name = "conv1D_lstm_train_output.csv"
train_output_path = os.path.join(save_path, data_params['city'], file_name)

cnn_lstm_model = Conv1D_LSTM(
    conv_filters=conv1d_lstm_model_params['filters'],
    conv_kernel_size=conv1d_lstm_model_params['kernel_size'],
    lstm_layers=conv1d_lstm_model_params['lstm_layers'],
    lstm_units=conv1d_lstm_model_params['lstm_units'],
    dense_layers=conv1d_lstm_model_params['dense_layers'],
    dense_units=conv1d_lstm_model_params['dense_units'],
    input_shape=X_train.shape[1:],
    activations=conv1d_lstm_model_params['activ'],
    if_dropout=conv1d_lstm_model_params['dropout'],
    dropout_val=conv1d_lstm_model_params['dropout_val']
)

print(cnn_lstm_model.summary())

tracker = EmissionsTracker(output_dir=os.path.join(os.getcwd(), save_path, data_params['city']),
    output_file='CO2_detailed_output.csv', project_name='Conv1d-LSTM')
tracker.start()

CLSTM = conf_model(cnn_lstm_model, X_train, y_train, X_val, y_val, train_output_path, mtype="DL",
    epochs=conv1d_lstm_model_params['epochs'], batch_size=conv1d_lstm_model_params['batch_size'], B_ARG=B_ARG)

emissions = tracker.stop()

y_test = y_all[split_idx:].copy() 
Y_pred, y_pred_lower, y_pred_upper = CLSTM.predict(X_test, alpha=.1, y_true=y_test, s=None)
Y_pred = scaler_y.inverse_transform(Y_pred.reshape(-1, 1))
y_pred_lower = scaler_y.inverse_transform(y_pred_lower.reshape(-1, 1))
y_pred_upper = scaler_y.inverse_transform(y_pred_upper.reshape(-1, 1))
y_test_inv   = scaler_y.inverse_transform(y_test.reshape(-1, 1))

conv1D_lstm_results = evaluate_preds(y_test_inv  , Y_pred, y_pred_lower, y_pred_upper)
plot_predicted_interval_dl(y_test_inv  , Y_pred, y_pred_lower, y_pred_upper, dates=list(dataset_test.index), title="CONV1D-LSTM")
plt.savefig(save_path + dirs[0] + 'clstm_model_with_lags_test_part_cf.png')

y_train = y_train[:val_split_idx]
Y_train_pred, y_train_pred_lower, y_train_pred_upper = CLSTM.predict(X_train, alpha=.1, y_true=y_train, s=None)
Y_train_pred = scaler_y.inverse_transform(Y_train_pred.reshape(-1, 1))
y_train_pred_lower = scaler_y.inverse_transform(y_train_pred_lower.reshape(-1, 1))
y_train_pred_upper = scaler_y.inverse_transform(y_train_pred_upper.reshape(-1, 1))
y_train_inv = scaler_y.inverse_transform(y_train.reshape(-1, 1))

plot_result(y_train_inv, y_test_inv  , Y_train_pred, Y_pred, dates=list(dataset.index), title='CONV1D-LSTM')
emissions = tracker.stop()
conv1D_lstm_results['CO2_EMISSIONS'] = emissions

models_predictions['Conv1D-LSTM'] = Y_pred

print("conv1D_lstm_results: ", conv1D_lstm_results)


# """Comparaison"""
models_results = pd.DataFrame({
    "Linear Regression" : linear_lr_results,
    "Ridge": ridge_results,
    "Random Forest" : rforest_results,
    "XGBoost" : xgboost_results,
    "LightGBM" : lightgbm_results,
    "SVR" : svr_results,
    "Conv1D": conv1D_results,
    "LSTM" : lstm_results,
    "GRU" : gru_results,
    "Conv1D-LSTM": conv1D_lstm_results
}).T

print_banner("Models results")
print(models_results)

# Create Taylor diagram
title = f"{data_params['city']} - Taylor Diagram"
fig, stats_summary = plot_taylor_diagram(models_predictions, y_test_inv, title)
plt.savefig(f"{save_path}{dirs[0]}{data_params['city']}_Taylor Diagram.png")

models_results.to_excel(save_path + dirs[2]+'Comparaison_ALL_Models.xlsx')
models_results.plot(figsize=(10, 7), kind="bar", rot = 20.0);
plt.savefig(save_path + dirs[0] + 'Comparaison_ALL_Models.png')

borda_count_all = borda_count_ranking(models_results)
borda_count_all.to_excel(save_path + dirs[2]+'BordaCount_ALL_Models.xlsx')

print_banner("Borda Count results")
print(borda_count_all)    

