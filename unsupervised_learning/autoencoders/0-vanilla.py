#!/usr/bin/env python3


"""create an autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """function to create an autoencoder"""
    """
    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for
        each hidden layer in the encoder respectively
    return encoder , decoder, auto
    """
    input_img = keras.Input(shape=(input_dims,))
    encoded = input_img
    for node in hidden_layers:
        encoded = keras.layers.Dense(node, activation='relu')(encoded)
    latent = keras.layers.Dense(latent_dims, activation='relu')(encoded)
    decoded = latent
    for node in reversed(hidden_layers):
        decoded = keras.layers.Dense(node, activation='relu')(decoded)
    output_img = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    encoder = keras.Model(input_img, latent)
    decoder = keras.Model(latent, output_img)
    autoencoder = keras.Model(input_img, decoder(encoder(input_img)))
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, autoencoder
