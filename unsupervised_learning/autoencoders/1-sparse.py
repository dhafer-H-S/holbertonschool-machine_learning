#!/usr/bin/env python3

"""a function that creates a sparse autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    -input_dims is an integer containing the dimensions of the model input
    -hidden_layers is a list containing the number of nodes for
        each hidden layer in the encoder respectively
    -latent_dims is an integer containing the dimension of
        the latent space representation
    -lambtha is the regularization parameter used for L1
        regularization on the encoded output

    return encoder , decoder, auto
    """
    inputs = keras.Input(shape=(input_dims,))
    encoded = inputs
    regularizers = keras.regularizers
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)
    latent = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=regularizers.l1(lambtha))(encoded)

    decoded = latent
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)
    output_img = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    encoder = keras.Model(inputs, latent)
    decoder = keras.Model(latent, output_img)
    autoencoder = keras.Model(inputs, decoder(encoder(inputs)))
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, autoencoder
