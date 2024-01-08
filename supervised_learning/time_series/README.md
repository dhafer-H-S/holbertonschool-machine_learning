Holberton school compuse of tunisia project : Bitcoin Price Forecasting using RNNs

Overview :

Bitcoin (BTC) has gained significant attention, especially after its price peak in 2018. Many individuals have sought ways to predict its value for potential financial gains. In this project, we aim to leverage Recurrent Neural Networks (RNNs) to forecast the closing value of BTC using the Coinbase and Bitstamp datasets. The primary focus will be on creating, training, and validating a Keras model for this forecasting task.


Data set type:
difference between coinbase and bitstamp
comparing Coinbase vs Bitstamp, the bigger active user base is gathered by Coinbase with around 108M users. Whereas Bitstamp has around 4M active users. If we look at the cryptocurrencies that are accepted by these exchanges, we can see that Coinbase has a higher number of acceptable crypto than Bitstamp.

Task Description :

Script: forecast_btc.py

Objective: Develop a script that utilizes the past 24 hours of BTC data to predict the value of BTC at the close of the following hour.
Datasets: Utilize the Coinbase and Bitstamp datasets, where each row represents a 60-second time window with various data points, such as open price, high price, low price, close price, BTC transacted, USD transacted, and volume-weighted average price.

Model Requirements:

Architecture: Implement an RNN architecture of your choosing for the model.

Cost Function: Use Mean Squared Error (MSE) as the cost function.

Data Feeding: Utilize tf.data.Dataset to feed data to the model.

Script: preprocess_data.py

Objective: Create a script to preprocess the raw dataset before training the model.

Keep only relevant features (start time, close price, etc.).
Rescale the data, usually using Min-Max scaling.
Create sequences of the past 24 hours of BTC data for each target (close price at the end of the hour).


Considerations:
Data Points Relevance: Evaluate the relevance of all data points in the dataset.
Feature Selection: Identify and select the most useful data features for the forecasting task.
Data Rescaling: Consider whether rescaling the data is necessary for better model performance.
Time Window Relevance: Assess the significance of the current time window for the forecasting task.
Save Preprocessed Data: Determine an appropriate method for saving the preprocessed data.
Usage Instructions
Clone the repository:


Run preprocess_data.py to preprocess the dataset:

Run forecast_btc.py to create, train, and validate the BTC forecasting model:

Additional Notes
Please refer to the documentation within each script for specific parameters and options.
Ensure that you have the necessary dependencies installed, including TensorFlow and Keras.
Feel free to experiment with different RNN architectures to optimize the model for forecasting BTC prices.
Happy forecasting!