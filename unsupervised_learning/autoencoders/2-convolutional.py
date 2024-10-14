#!/usr/bin/env python3
"""A function that creates a convolutional autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.

    Args:
        input_dims (tuple): Dimensions of the model input.
        filters (list): List containing the number of filters for each
        convolutional layer in the encoder.
        latent_dims (tuple): Dimensions of the latent space representation.

    Returns:
        tuple: encoder, decoder, autoencoder models.
    """
    """Define the input layer with the specified input dimensions"""
    input_lay = keras.Input(shape=input_dims)
    x = input_lay

    """Build the encoder"""
    for filter in filters:
        x = keras.layers.Conv2D(
            filters=filter, kernel_size=(
                3, 3), padding='same', activation='relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    """Output of the encoder"""
    encoder_out = x
    encoder = keras.models.Model(input_lay, encoder_out, name='encoder')

    """
    Define the input for the decoder with
    the shape of the encoder's output
    """
    decoder_inp = keras.layers.Input(shape=encoder_out.shape[1:])
    x = decoder_inp

    """Build the decoder"""
    for i, filter in enumerate(reversed(filters)):
        if i == len(filters) - 1:
            x = keras.layers.Conv2D(
                filters=filter, kernel_size=(
                    3, 3), padding='valid', activation='relu')(x)
        else:
            x = keras.layers.Conv2D(
                filters=filter, kernel_size=(
                    3, 3), padding='same', activation='relu')(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)

    """Final convolutional layer to match the input dimensions"""
    x = keras.layers.Conv2D(
        filters=input_dims[-1], kernel_size=(3, 3),
        padding='same', activation='sigmoid')(x)

    """Output of the decoder"""
    decoder_output = x
    decoder = keras.models.Model(decoder_inp, decoder_output, name='decoder')

    """Combine the encoder and decoder into the autoencoder model"""
    autoencoder = keras.models.Model(
        inputs=input_lay,
        outputs=decoder(encoder(input_lay)),
        name='autoencoder')

    """Compile the autoencoder model"""
    autoencoder.compile(optimizer='adam', loss="binary_crossentropy")

    return encoder, decoder, autoencoder
