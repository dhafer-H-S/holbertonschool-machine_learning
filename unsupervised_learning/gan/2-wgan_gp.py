#!/usr/bin/env python3
"""
Implement a Wasserstein GAN with Gradient Penalty (WGAN-GP).
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_GP(keras.Model):
    """Class for Wasserstein GANs with gradient penalty."""

    def __init__(
            self,
            generator,
            discriminator,
            latent_generator,
            real_examples,
            batch_size=200,
            disc_iter=2,
            learning_rate=.005,
            lambda_gp=10):
        """
        Initialize the WGAN_GP model.

        Args:
            generator (keras.Model): The generator model.
            discriminator (keras.Model): The discriminator model.
            latent_generator (function): Function to generate latent vectors.
            real_examples (np.array): Array of real examples.
            batch_size (int): Batch size for training.
            disc_iter (int): Number of discriminator iterations per
            generator iteration.
            learning_rate (float): Learning rate for the optimizers.
            lambda_gp (float): Gradient penalty coefficient.
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate
        self.beta_1 = .3
        self.beta_2 = .9
        self.lambda_gp = lambda_gp
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        """Define the generator loss and optimizer"""
        self.generator.loss = lambda x: - \
            tf.math.reduce_mean(self.discriminator(x))
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss)

        """Define the discriminator loss and optimizer"""
        self.discriminator.loss = lambda x, y: tf.math.reduce_mean(
            self.discriminator(y)) - tf.math.reduce_mean(self.discriminator(x))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """
        Generate fake samples of the specified size.

        Args:
            size (int, optional): Number of samples to generate.
            Defaults to batch_size.
            training (bool, optional): Whether the model is in training mode.
            Defaults to False.

        Returns:
            np.array: Array of generated fake samples.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """
        Generate real samples of the specified size.

        Args:
            size (int): Number of samples to generate. Defaults to batch_size.

        Returns:
            np.array: Array of real samples.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def get_interpolated_sample(self, real_sample, fake_sample):
        """
        Generate interpolated samples between real and fake samples.

        Args:
            real_sample (np.array): Array of real samples.
            fake_sample (np.array): Array of fake samples.

        Returns:
            np.array: Array of interpolated samples.
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """
        Compute the gradient penalty for the interpolated samples.

        Args:
            interpolated_sample (np.array): Array of interpolated samples.

        Returns:
            float: Gradient penalty.
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, useless_argument):
        """
        Perform one training step.

        Args:
            useless_argument: Placeholder argument for compatibility.

        Returns:
            dict: Dictionary containing discriminator and generator
            losses and gradient penalty.
        """
        for _ in range(self.disc_iter):
            with tf.GradientTape() as disc_tape:
                # Get real and fake samples
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)
                interpolated_samples = self.get_interpolated_sample(
                    real_samples, fake_samples)

                # Compute the loss for the discriminator
                discr_loss = self.discriminator.loss(
                    real_samples, fake_samples)
                gp = self.gradient_penalty(interpolated_samples)
                new_discr_loss = discr_loss + self.lambda_gp * gp

            # Apply gradient descent to the discriminator
            gradients_of_discriminator = disc_tape.gradient(
                new_discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(gradients_of_discriminator,
                    self.discriminator.trainable_variables))

        # Compute the loss for the generator
        with tf.GradientTape() as gen_tape:
            fake_samples = self.get_fake_sample(training=True)
            gen_loss = self.generator.loss(fake_samples)

        # Apply gradient descent to the generator
        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}
