#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_clip(keras.Model):
    """class Wasserstein GANs"""

    def __init__(
            self,
            generator,
            discriminator,
            latent_generator,
            real_examples,
            batch_size=200,
            disc_iter=2,
            learning_rate=.005):
        """ initiation function """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate
        self.beta_1 = .5
        self.beta_2 = .9

        """Define the generator loss and optimizer"""
        self.generator.loss = lambda x: - \
            tf.math.reduce_mean(self.discriminator(x))
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss)

        """Define the discriminator loss and optimizer"""
        self.discriminator.loss = lambda x, y: tf.math.reduce_mean(
            self.discriminator(y)) - tf.math.reduce_mean(self.discriminator(x))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss)

    """Generator of fake samples of size batch_size"""

    def get_fake_sample(self, size=None, training=False):
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    """Generator of real samples of size batch_size"""

    def get_real_sample(self, size=None):
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    """Overloading train_step()"""

    def train_step(self, useless_argument):
        for _ in range(self.disc_iter):
            with tf.GradientTape() as disc_tape:
                """Get real and fake samples"""
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)

                """Compute the loss for the discriminator"""
                discr_loss = self.discriminator.loss(
                    real_samples, fake_samples)

            """Apply gradient descent to the discriminator"""
            gradients_of_discriminator = disc_tape.gradient(
                discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables))

            """Clip the weights of the discriminator between -1 and 1"""
            for weight in self.discriminator.trainable_variables:
                weight.assign(tf.clip_by_value(weight, -1.0, 1.0))

        """Compute the loss for the generator"""
        with tf.GradientTape() as gen_tape:
            fake_samples = self.get_fake_sample(training=True)
            gen_loss = self.generator.loss(fake_samples)

        """Apply gradient descent to the generator"""
        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
