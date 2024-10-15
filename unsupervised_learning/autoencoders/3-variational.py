#!/usr/bin/env python3
"""A function that creates a variational autoencoder"""
import tensorflow.keras as keras
import tensorflow as tf


def autoencoder(input_dim, hidden_units, latent_dim):
    """
    Builds a variational autoencoder (VAE) model.

    Args:
        input_dim (int): The dimensionality of the input data.
        hidden_units (list of int): A list containing the number of
        units for each hidden layer in the encoder and decoder.
        latent_dim (int): The dimensionality of the latent space.

    Returns:
        tuple: A tuple containing the encoder model, decoder model,
        and the VAE model.
            - encoder (keras.Model): The encoder part of the VAE.
            - decoder (keras.Model): The decoder part of the VAE.
            - vae (keras.Model): The complete VAE model.

    The VAE model is trained to minimize the reconstruction loss and
    the KL divergence loss.
    """
    """Define the input layer with the specified input dimension"""
    input_layer = keras.Input(shape=(input_dim,))
    hidden_layer = input_layer

    """Build the encoder with the specified hidden layers"""
    for units in hidden_units:
        hidden_layer = keras.layers.Dense(
            units, activation='relu')(hidden_layer)

    """Define the mean and log variance layers for the latent space"""
    mean_layer = keras.layers.Dense(latent_dim, activation=None)(hidden_layer)
    log_var_layer = keras.layers.Dense(
        latent_dim, activation=None)(hidden_layer)

    def reparametrize(args):
        """Reparameterization method"""
        mean, log_var = args
        batch_size = tf.shape(mean)[0]
        latent_size = tf.shape(mean)[1]
        noise = tf.random.normal(shape=(batch_size, latent_size))
        return mean + tf.exp(0.5 * log_var) * noise

    """Sample from the latent space using the reparameterization method"""
    latent_sample = keras.layers.Lambda(
        reparametrize)([mean_layer, log_var_layer])

    """Define the encoder model"""
    encoder_model = keras.Model(
        inputs=input_layer,
        outputs=[latent_sample, mean_layer, log_var_layer],
        name='encoder'
    )

    """Define the input for the decoder with the shape of the latent space"""
    decoder_input = keras.Input(shape=(latent_dim,))
    hidden_decoded = decoder_input

    """Build the decoder with the hidden layers in reverse order"""
    for units in reversed(hidden_units):
        hidden_decoded = keras.layers.Dense(
            units, activation='relu')(hidden_decoded)

    """Final layer to match the input dimension with sigmoid activation"""
    output_layer = keras.layers.Dense(
        input_dim, activation='sigmoid')(hidden_decoded)

    """Define the decoder model"""
    decoder_model = keras.Model(
        inputs=decoder_input,
        outputs=output_layer,
        name='decoder'
    )

    """Combine the encoder and decoder into the autoencoder model"""
    final_output = decoder_model(encoder_model(input_layer)[0])
    vae = keras.Model(inputs=input_layer, outputs=final_output, name='vae')

    """Calculate the reconstruction loss"""
    reconstruction_loss = keras.losses.binary_crossentropy(
        input_layer, final_output)
    reconstruction_loss *= input_dim

    """Define KL divergence loss"""
    kl_divergence_loss = 1 + log_var_layer - \
        tf.square(mean_layer) - tf.exp(log_var_layer)
    kl_divergence_loss = tf.reduce_sum(kl_divergence_loss, axis=-1)
    kl_divergence_loss *= -0.5

    """Combine the reconstruction loss and KL divergence loss"""
    total_loss = tf.reduce_mean(reconstruction_loss + kl_divergence_loss)
    vae.add_loss(total_loss)

    """Compile the autoencoder model"""
    vae.compile(optimizer='adam')

    return encoder_model, decoder_model, vae
