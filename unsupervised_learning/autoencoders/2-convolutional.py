#!/usr/bin/env python3

"""A function that creates a convolutional autoencoder"""

import tensorflow.keras as keras

def autoencoder(input_dims, filters, latent_dims):
    """
    - input_dims is a tuple of integers containing the dimensions of the model input
    - filters is a list containing the number of filters for each convolutional layer in the encoder, respectively
    - latent_dims is a tuple of integers containing the dimensions of the latent space representation

    Returns: encoder, decoder, autoencoder
    """
    # Input layer
    inputs = keras.Input(shape=input_dims)
    
    # Encoder
    encoded = inputs
    for f in filters:
        encoded = keras.layers.Conv2D(f, (3, 3), padding='same', activation='relu')(encoded)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
    
    # Latent space representation
    latent = keras.layers.Conv2D(latent_dims[2], (3, 3), padding='same', activation='relu')(encoded)
    
    # Decoder
    decoded = latent
    for f in reversed(filters):
        decoded = keras.layers.Conv2D(f, (3, 3), padding='same', activation='relu')(decoded)
        decoded = keras.layers.UpSampling2D((2, 2))(decoded)
    
    # Adjust the final upsampling to match the input dimensions
    decoded = keras.layers.Conv2D(filters[0], (3, 3), padding='same', activation='relu')(decoded)
    decoded = keras.layers.Conv2D(input_dims[-1], (3, 3), padding='same', activation='sigmoid')(decoded)
    
    # Ensure the output dimensions match the input dimensions
    output_img = keras.layers.Cropping2D(((2, 2), (2, 2)))(decoded)

    # Models
    encoder = keras.Model(inputs, latent)
    decoder = keras.Model(latent, output_img)
    autoencoder = keras.Model(inputs, decoder(encoder(inputs)))

    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder