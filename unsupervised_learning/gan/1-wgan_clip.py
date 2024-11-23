#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class WGAN_clip(keras.Model):
    """
    Wasserstein Generative Adversarial Network with weight clipping (WGAN-clip).

    Attributes:
        generator (keras.Model): The generator model.
        discriminator (keras.Model): The discriminator model.
        latent_generator (callable): Function to generate latent space vectors.
        real_examples (tf.Tensor): Tensor containing real examples.
        batch_size (int): Batch size for training. Default is 200.
        disc_iter (int): Number of discriminator iterations per generator iteration. Default is 2.
        learning_rate (float): Learning rate for the optimizers. Default is 0.005.
        clip_value (float): Value to clip the discriminator weights. Default is 0.01.
        beta_1 (float): Beta_1 parameter for the Adam optimizer. Default is 0.5.
        beta_2 (float): Beta_2 parameter for the Adam optimizer. Default is 0.9.

    Methods:
        get_fake_sample(size=None, training=False):
            Generates fake samples using the generator model.
        
        get_real_sample(size=None):
            Retrieves real samples from the provided real examples.
        
        train_step(useless_argument):
            Performs one training step, updating both the discriminator and generator.
    """
    def __init__(self, generator, discriminator, latent_generator, real_examples, batch_size=200, disc_iter=2, learning_rate=.005, clip_value=0.01):
        """initiate function"""
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
        self.clip_value = clip_value

        """Define the generator loss:"""
        self.generator.loss = lambda x: -tf.math.reduce_mean(self.discriminator(x))

        """Define the discriminator loss:"""
        self.discriminator.loss = lambda x, y: tf.math.reduce_mean(self.discriminator(y)) - tf.math.reduce_mean(self.discriminator(x))

        """Define optimizers for both models"""
        self.generator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.discriminator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)

        """Compile the models"""
        self.generator.compile(optimizer=self.generator.optimizer, loss=self.generator.loss)
        self.discriminator.compile(optimizer=self.discriminator.optimizer, loss=self.discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """fake data function"""
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """fake data function"""
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """train function"""
        for _ in range(self.disc_iter):
            with tf.GradientTape() as disc_tape:
                """Get real and fake samples"""
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)

                """Calculate discriminator loss"""
                discr_loss = self.discriminator.loss(real_samples, fake_samples)

            """Apply gradients to discriminator"""
            gradients_of_discriminator = disc_tape.gradient(discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

            """Clip discriminator weights"""
            for weight in self.discriminator.trainable_variables:
                weight.assign(tf.clip_by_value(weight, -self.clip_value, self.clip_value))

        """Train Generator"""
        with tf.GradientTape() as gen_tape:
            """Get fake samples"""
            fake_samples = self.get_fake_sample(training=True)

            """Calculate generator loss"""
            gen_loss = self.generator.loss(fake_samples)

        """Apply gradients for generator"""
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}