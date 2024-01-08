#!/usr/bin/env python3


# preprocess_data.py
import pandas as pd
import numpy as np
import tensorflow as tf

def preprocessing(csv_paths):
    dfs = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        # drop nan
        df = df.dropna()
        # take the last 2 years data
        df = df[-(730 * 24 * 60):]
        # encode the date
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
        # make date as index
        df = df.set_index('Date')
        df.drop(['Timestamp'], axis=1, inplace=True)
        dfs.append(df)

    # concatenate all dataframes
    df = pd.concat(dfs)
    print(df.head())

    # you should always split your data into training, validation, testing
    # (usually 80%, 10%, 10%)
    df_train = df[:int(len(df)*80/100)]
    df_valid = df[int(len(df_train)):int(len(df)*90/100)]
    df_test = df[int(-(len(df)*10/100)):]

    # normalize data
    train_mean = df_train.mean()
    train_std = df_train.std()
    x_train = (df_train - train_mean) / train_std
    x_valid = (df_valid - train_mean) / train_std
    x_test = (df_test - train_mean) / train_std
    return x_train, x_valid, x_test

# Call the function with the paths to your CSV files
x_train, x_valid, x_test = preprocessing(['bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', 'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'])
