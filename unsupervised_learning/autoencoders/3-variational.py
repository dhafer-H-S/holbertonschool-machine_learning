#!/usr/bin/env python3
"""A function that creates a variational autoencoder"""
import tensorflow.keras as keras
import tensorflow as tf


def autoencoder(input_dim, hidden_units, latent_dim):
    """Returns: encoder, decoder, vae"""
    input_layer = keras.Input(shape=(input_dim,))
    hidden_layer = input_layer

    for units in hidden_units:
        hidden_layer = keras.layers.Dense(
            units, activation='relu')(hidden_layer)
    mean_layer = keras.layers.Dense(latent_dim, activation=None)(hidden_layer)
    log_var_layer = keras.layers.Dense(
        latent_dim, activation=None)(hidden_layer)

    def reparametrize(args):
        """Reparameterization trick"""
        mean, log_var = args
        batch_size = tf.shape(mean)[0]
        latent_size = tf.shape(mean)[1]
        noise = tf.random.normal(shape=(batch_size, latent_size))
        return mean + tf.exp(0.5 * log_var) * noise

    latent_sample = keras.layers.Lambda(
        reparametrize)([mean_layer, log_var_layer])
    encoder_model = keras.Model(
        inputs=input_layer,
        outputs=[latent_sample, mean_layer, log_var_layer],
        name='encoder'
    )
    decoder_input = keras.Input(shape=(latent_dim,))
    hidden_decoded = decoder_input
    for units in reversed(hidden_units):
        hidden_decoded = keras.layers.Dense(
            units, activation='relu')(hidden_decoded)
    output_layer = keras.layers.Dense(
        input_dim, activation='sigmoid')(hidden_decoded)
    decoder_model = keras.Model(
        inputs=decoder_input,
        outputs=output_layer,
        name='decoder'
    )
    final_output = decoder_model(encoder_model(input_layer)[0])
    vae = keras.Model(inputs=input_layer, outputs=final_output, name='vae')
    reconstruction_loss = keras.losses.binary_crossentropy(
        input_layer, final_output)
    reconstruction_loss *= input_dim
    kl_divergence_loss = 1 + log_var_layer - \
        tf.square(mean_layer) - tf.exp(log_var_layer)
    kl_divergence_loss = tf.reduce_sum(kl_divergence_loss, axis=-1)
    kl_divergence_loss *= -0.5
    total_loss = tf.reduce_mean(reconstruction_loss + kl_divergence_loss)
    vae.add_loss(total_loss)
    vae.compile(optimizer='adam')
    return encoder_model, decoder_model, vae
