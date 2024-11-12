#!/usr/bin/env python3
"""
simple gan
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class Simple_GAN(keras.Model):
    """
    simple gan
    """
    def __init__(self, generator, discriminator,
                 latent_generator, real_examples,
                 batch_size=200, disc_iter=2, learning_rate=.005):
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

        """define the generator loss and optimizer: """
        self.generator.loss =\
            lambda x: tf.keras.losses.MeanSquaredError()(x,
                                                         tf.ones(x.shape))
        self.generator.optimizer =\
            keras.optimizers.Adam(learning_rate=self.learning_rate,
                                  beta_1=self.beta_1, beta_2=self.beta_2)
        self.generator.compile(optimizer=self.generator.optimizer,
                               loss=self.generator.loss)

        """define the discriminator loss and optimizer: """
        self.discriminator.loss =\
            lambda x, y: (tf.keras.losses.MeanSquaredError()
                          (x, tf.ones(x.shape))
                          + tf.keras.losses.MeanSquaredError()
                          (y, -1*tf.ones(y.shape)))
        self.discriminator.optimizer =\
            keras.optimizers.Adam(learning_rate=self.learning_rate,
                                  beta_1=self.beta_1,
                                  beta_2=self.beta_2)
        self.discriminator.compile(optimizer=self.discriminator.optimizer,
                                   loss=self.discriminator.loss)

    """generator of real samples of size batch_size """
    def get_fake_sample(self, size=None, training=False):
        """
        Some documentation
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size),
                              training=training)

    """generator of fake samples of size batch_size """
    def get_real_sample(self, size=None):
        """
        Some documentation
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    """overloading train_step() """
    def train_step(self, useless_argument):
        """
        This is the training function
        """
        discr_loss = 0
        for _ in range(self.disc_iter):
            with tf.GradientTape() as disc_tape:
                """get a real sample """
                real_sample = self.get_real_sample()
                """get a fake sample """
                fake_sample = self.get_fake_sample(training=True)
                discr_loss = self.discriminator.\
                    loss(self.discriminator(real_sample,
                                            training=True),
                         self.discriminator(fake_sample,
                                            training=True))
            """apply gradient descent once to the discriminator """
            gradients_of_discriminator =\
                disc_tape.gradient(discr_loss,
                                   self.discriminator.trainable_variables)
            self.discriminator.\
                optimizer.apply_gradients(zip(gradients_of_discriminator,
                                              self.discriminator.
                                              trainable_variables))

        with tf.GradientTape() as gen_tape:
            """get a fake sample """
            fake_sample = self.get_fake_sample(training=True)
            gen_loss = self.generator.loss(self.discriminator(fake_sample,
                                                              training=True))
        """ apply gradient descent to the generator"""
        gradients_of_generator =\
            gen_tape.gradient(gen_loss,
                              self.generator.trainable_variables)
        self.generator.optimizer.\
            apply_gradients(zip(gradients_of_generator,
                                self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
