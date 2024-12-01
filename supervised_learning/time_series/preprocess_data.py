#!/usr/bin/env python3

import pandas as pd
import numpy as np

def preprocessing(csv_paths, output_paths):
    dfs = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        # Forward fill to handle missing values
        df = df.ffill()
        # Convert the timestamp to datetime format
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
        # Filter data to include only entries from the last two years
        df = df[df['Date'] >= '2017-01-01']
        # Drop the timestamp column as it's no longer needed
        df.drop(['Timestamp'], axis=1, inplace=True)
        # Set the date as the index
        df = df.set_index('Date')
        dfs.append(df)

    # Concatenate all dataframes
    df = pd.concat(dfs)

    # Split the data into training, validation, and testing sets (80%, 10%, 10%)
    df_train = df[:int(len(df) * 80 / 100)]
    df_valid = df[int(len(df_train)):int(len(df) * 90 / 100)]
    df_test = df[int(-(len(df) * 10 / 100)):]

    # Normalize the data
    train_mean = df_train.mean()
    train_std = df_train.std()
    df_train = (df_train - train_mean) / train_std
    df_valid = (df_valid - train_mean) / train_std
    df_test = (df_test - train_mean) / train_std

    # Save the preprocessed data to new CSV files
    df_train.to_csv(output_paths[0])
    df_valid.to_csv(output_paths[1])
    df_test.to_csv(output_paths[2])

    return df_train, df_valid, df_test

# Call the function with the paths to your CSV files and output files
df_train, df_valid, df_test = preprocessing(
    ['bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', 'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'],
    ['x_train.csv', 'x_valid.csv', 'x_test.csv']
)
