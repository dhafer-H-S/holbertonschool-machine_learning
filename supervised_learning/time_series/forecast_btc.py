#!/usr/bin/env python3

from tensorflow import keras as k
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import tensorflow as tf
from preprocess_data import preprocessing

class WindowGenerator:
    """WindowGenerator Class"""

    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        """
        Initialize the WindowGenerator class.

        Args:
            input_width (int): Width of the input window.
            label_width (int): Width of the label window.
            shift (int): Number of steps to shift the label window.
            train_df (pandas.DataFrame): Training data.
            val_df (pandas.DataFrame): Validation data.
            test_df (pandas.DataFrame): Test data.
            label_columns (list): List of column names to be used as labels.
        """

        super().__init__()
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        # Create labels
        self.train_labels = self.train_df[self.label_columns]
        self.val_labels = self.val_df[self.label_columns]
        self.test_labels = self.test_df[self.label_columns]
    def split_window(self, features):
        """
        Split the window into inputs and labels.

        Args:
            features (numpy.ndarray): Window features.

        Returns:
            inputs (numpy.ndarray): Input data.
            labels (numpy.ndarray): Label data.
        """

        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]]
                 for name in self.label_columns],
                axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='Close', max_subplots=3):
        """
        Plot the input and label data.

        Args:
            model (tf.keras.Model): Trained model for making predictions.
            plot_col (str): Column name to plot.
            max_subplots (int): Maximum number of subplots to display.
        """

        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_idx = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n + 1)
            plt.ylabel('{} [normed]'.format(plot_col))
            plt.plot(self.input_indices, inputs[n, :, plot_col_idx],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_idx = self.label_columns_indices.get(plot_col,
                                                               None)
            else:
                label_col_idx = plot_col_idx

            if label_col_idx is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_idx],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                prediction = model(inputs)
                plt.scatter(self.label_indices,
                            prediction[n, :, label_col_idx],
                            marker='X', edgecolors='k',
                            label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

    def make_dataset(self, data):
        """
        Convert the data to a tf.dataset.

        Args:
            data (numpy.ndarray): Data to be converted.

        Returns:
            ds (tf.data.Dataset): Converted dataset.
        """

        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32, )

        ds = ds.map(self.split_window)

        return ds
    @property
    def train_dataset(self):
        """
        Create the training dataset.

        Returns:
            ds (tf.data.Dataset): Training dataset.
        """

        return self.make_dataset(self.train_df)
    @property
    def validation_dataset(self):
        """
        Create the validation dataset.

        Returns:
            ds (tf.data.Dataset): Validation dataset.
        """

        return self.make_dataset(self.val_df)
    @property
    def test_dataset(self):
        """
        Create the test dataset.

        Returns:
            ds (tf.data.Dataset): Test dataset.
        """

        return self.make_dataset(self.test_df)
    
# class Baseline(tf.keras.Model):
#     """
#     Baseline class
#     """

#     def __init__(self, label_index=None):
#         """
#         Class constructor
#         Argumetns:
#          - label_index
#         """

#         super().__init__()
#         self.label_index = label_index

#     def call(self, inputs):
#         """
#         call
#         Argumetns:
#          - param inputs
#         """
#         if self.label_index is None:
#             return inputs
#         result = inputs[:, :, self.label_index]
#         return result[:, :, tf.newaxis]



def compile_and_fit(model, window, patience=2, epochs=500):
    """compile and fit model"""
    e_s = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=patience,
                                           mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    history = model.fit(window.train_dataset, validation_data=(window.validation_dataset), epochs=epochs, callbacks=[e_s])

    print(model.summary())


    return history

def forecasting(train, validation, test):
    """
    Function for forecasting model of the BTC price
    Arguments:
     - train is the train values
     - validation is the validation values
     - test is the test values
    """

    window = WindowGenerator(input_width=24, label_width=24, shift=1,
                             train_df=train, val_df=validation, test_df=test,
                             label_columns=['Close'])
    column_indices = window.column_indices
    print(window)

    val_performance = {}
    performance = {}

    # baseline = Baseline(label_index=column_indices['Close'])

    # baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
    #                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

    lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dense(units=1)])
    history = compile_and_fit(lstm_model, window)

    val_performance['LSTM'] = lstm_model.evaluate(window.val)
    performance['LSTM'] = lstm_model.evaluate(window.test, verbose=0)
    window.plot(lstm_model)

# Call the function with the paths to your CSV files
x_train, x_valid, x_test = preprocessing(['bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', 'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'])
forecasting(x_train, x_valid, x_test)