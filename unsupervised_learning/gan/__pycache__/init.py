import keras
import tensorflow as tf    


    def __init( self, generator , discriminator , latent_generator, real_examples, 
                batch_size=200, disc_iter=2, learning_rate=.005):
                
        super().__init__()                         # run the __init__ of Keras.Model first. 
        self.latent_generator = latent_generator
        self.real_examples    = real_examples
        self.generator        = generator
        self.discriminator    = discriminator
        self.batch_size       = batch_size
        self.disc_iter        = disc_iter
        
        self.learning_rate=learning_rate
        self.beta1=.5                               # standard value, but can be changed if necessary
        self.beta2=.9                               # standard value, but can be changed if necessary
        
        # define the generator loss and optimizer:
        self.generator.loss      = lambda x : tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape))
        self.generator.optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
        self.generator.compile(optimizer=generator.optimizer , loss=generator.loss )
        
        # define the discriminator loss and optimizer:
        self.discriminator.loss      = lambda x , y : tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape)) + tf.keras.losses.MeanSquaredError()(y, -1*tf.ones(y.shape))
        self.discriminator.optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
        self.discriminator.compile(optimizer=discriminator.optimizer , loss=discriminator.loss )
    
    def get_fake_sample(self, training=False):
        self.generator(self.latent_generator(self.batch_size), training=training)

    def get_real_sample(self):
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices  = tf.random.shuffle(sorted_indices)[:self.batch_size]
        return tf.gather(self.real_examples, random_indices)