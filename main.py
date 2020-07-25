# -----------------------------------------------------------
# demonstrates how to normalize data within each operating regime with multi-layer perceptron
#
# (C) 2020 Marcia Baptista, Lisbon, Portugal
# Released under GNU Public License (GPL)
# email marcia.lbaptista@gmail.com
# -----------------------------------------------------------
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from keras import optimizers
from sklearn.cluster import KMeans
from keras.layers import Dense
from keras.models import Sequential
from keras import initializers
from keras import backend as K

###########################################
#
# Auxiliary reading functions
#
###########################################

np.random.seed(7)

def read_pandas_array(pd_array):
    '''
    joins the data of different C-MAPSS files in a single dataframe
    :param pd_array: list of pandas array
    :return: dataframe
    '''
    frames = []
    for i in range(len(pd_array)):
        frames.append(pd_array[i])
    return pd.concat(frames, ignore_index=True)

###########################################
#
# Features
#
###########################################


# features of C-MAPSS different data sets
feature_names = ['unit_number', 'time', 'altitude', 'mach_number', 'throttle_resolver_angle',
                 'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi',
                 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']

# features that correspond to sensors
sensor_names = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi',
                'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']

# features that correspond to operational conditions
op_condition_features = ['altitude', 'mach_number', 'throttle_resolver_angle']

nonflat_sensor_names = ['T24', 'T30', 'T50', 'phi', 'htBleed']


def custom_objective(y_true, y_pred):
    return K.square(y_pred)


def build_model(learning_rate=10e-9, min_val=-0.01, max_val=0, nr_regimes=6):
    model = Sequential()
    N = nr_regimes + 1
    init = initializers.RandomUniform(minval=-0.01, maxval=0, seed=None)
    model.add(Dense(1, kernel_initializer=init, activation="linear"))

    sgd = optimizers.SGD(lr=learning_rate, decay=0, momentum=0)
    model.compile(loss=custom_objective, optimizer=sgd, metrics=['mean_squared_error'], shuffle=False)
    return model


def normalize_data(dataset_id, sensor_name, learning_rate, min_val, max_val, epochs, batch_size, visualize_normalization):

    if dataset_id == 2:
        train_data = [
            pd.read_csv('data/train_FD002.txt', sep='\s+', names=feature_names),
        ]
    else:
        train_data = [
            pd.read_csv('data/train_FD004.txt', sep='\s+', names=feature_names),
        ]

    df = read_pandas_array(train_data)

    # discover the clusters
    nr_regimes = 6
    k_means = KMeans(n_clusters=nr_regimes, random_state=0).fit(df.loc[:, op_condition_features])

    # normalize the data using standard deviation
    df[sensor_name + 'scaled'] = 0
    for regime in range(nr_regimes):
        signal_regime = df.loc[(k_means.labels_ == regime), sensor_name].values
        df.loc[k_means.labels_ == regime, sensor_name + 'scaled'] = (signal_regime - np.mean(signal_regime)) / np.std(
            signal_regime)

    # normalize the data using multi-layer perceptron
    max_signal = np.max(df.loc[:, [sensor_name]].values)
    max_signal = max_signal

    v = df.loc[(k_means.labels_ == 0), sensor_name].values
    temp = np.std(v)
    print("Std of %s" % temp)
    df['regime0'] = (k_means.labels_ == 0) * max_signal
    df['regime1'] = (k_means.labels_ == 1) * max_signal
    df['regime2'] = (k_means.labels_ == 2) * max_signal
    df['regime3'] = (k_means.labels_ == 3) * max_signal
    df['regime4'] = (k_means.labels_ == 4) * max_signal
    df['regime5'] = (k_means.labels_ == 5) * max_signal
    X = df

    model = build_model(learning_rate=learning_rate, min_val=min_val, max_val=max_val, nr_regimes=nr_regimes)
    input_of_model = np.array(X.loc[:, [sensor_name, 'regime0', 'regime1', 'regime2', 'regime3', 'regime4', 'regime5']])
    signal = np.array(X.loc[:, [sensor_name + 'scaled']])
    output_of_model = signal

    history = model.fit(input_of_model, output_of_model, epochs=epochs, batch_size=batch_size)

    layer_outputs = [layer.output for layer in model.layers[:]]
    predictions = model.predict(input_of_model)

    X[sensor_name + 'scaledMLP'] = predictions

    errors = []
    units = np.unique(X['unit_number'])
    for unit in units:
        predicted_signal = predictions[(X['unit_number'] == unit)]
        predicted_signal = [item for sublist in predicted_signal for item in sublist]

        scaled_signal = signal[(X['unit_number'] == unit)]
        scaled_signal = [item for sublist in scaled_signal for item in sublist]

        n_predicted_signal = (predicted_signal - np.min(predicted_signal)) / (np.max(predicted_signal) - np.min(predicted_signal))
        n_scaled_signal = (scaled_signal - np.min(scaled_signal)) / (np.max(scaled_signal) - np.min(scaled_signal))

        if visualize_normalization:
            plt.scatter(range(len(n_scaled_signal)), n_scaled_signal, c="red", label="Standard rule")
            plt.scatter(range(len(n_predicted_signal)), n_predicted_signal, c="black", label="MLP")

            plt.ylabel(sensor_name)
            plt.xlabel("Time")
            plt.legend()
            plt.tight_layout()
            plt.show()

        errors.append((100 * np.nanmean(np.abs(np.subtract(n_predicted_signal, n_scaled_signal)))))

    return np.mean(errors), predictions


mean_unit_error, normalized_sensor = normalize_data(dataset_id=2, sensor_name="phi", learning_rate=10e-9, min_val=-0.01, max_val=0, epochs=10, batch_size=32, visualize_normalization=True)

print(mean_unit_error)