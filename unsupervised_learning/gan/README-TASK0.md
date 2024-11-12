In this task we will define a class Simple_GAN and test it on some examples. The instruction text is quite long, since we explain the context in details, but the exercise in itself is short : it consists in filling in the train_step method.


The generator and the discriminator networks

We will assume the generator network, the discriminator network, and a generator of latent vectors are already fixed. For example, as the result of the following function :


def spheric_generator(nb_points, dim) :
    u=tf.random.normal(shape=(nb_points, dim))
    return u/tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(u),axis=[1])+10**-8),[nb_points,1])

def fully_connected_GenDiscr(gen_shape, real_examples, latent_type="normal" ) :
    
    #   Latent generator   
    if latent_type   == "uniform" :
        latent_generator  =  lambda k : tf.random.uniform(shape=(k, gen_shape[0]))
    elif latent_type == "normal" :
        latent_generator  =  lambda k : tf.random.normal(shape=(k, gen_shape[0])) 
    elif latent_type == "spheric" :
        latent_generator  = lambda k : spheric_generator(k,gen_shape[0]) 
    
    #   Generator  
    inputs     = keras.Input(shape=( gen_shape[0] , ))
    hidden     = keras.layers.Dense( gen_shape[1] , activation = 'tanh'    )(inputs)
    for i in range(2,len(gen_shape)-1) :
        hidden = keras.layers.Dense( gen_shape[i] , activation = 'tanh'    )(hidden)
    outputs    = keras.layers.Dense( gen_shape[-1], activation = 'sigmoid' )(hidden)
    generator  = keras.Model(inputs, outputs, name="generator")
    
    #   Discriminator     
    inputs        = keras.Input(shape=( gen_shape[-1], ))
    hidden        = keras.layers.Dense( gen_shape[-2],   activation = 'tanh' )(inputs)
    for i in range(2,len(gen_shape)-1) :
        hidden    = keras.layers.Dense( gen_shape[-1*i], activation = 'tanh' )(hidden)
    outputs       = keras.layers.Dense( 1 ,              activation = 'tanh' )(hidden)
    discriminator = keras.Model(inputs, outputs, name="discriminator")
    
    return generator, discriminator, latent_generator
The code above produces two networks that are almost symmetric (the first layer of the generator and the last layer of the discriminator differ).

Note that the last layer of the discriminator has the sigmoid for activation function, thus takes values in 
, while all the others activation functions are the hyperbolic tangent, which takes values in 
.

For example :

generator, discriminator, latent_generator = fully_connected_GenDiscr([1,100,100,2], None)
print(generator.summary())
print(discriminator.summary())
produces :


Model: "generator"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 1)]               0         
_________________________________________________________________
dense (Dense)                (None, 100)               200       
_________________________________________________________________
dense_1 (Dense)              (None, 100)               10100     
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 202       
=================================================================
Total params: 10,502
Trainable params: 10,502
Non-trainable params: 0
_________________________________________________________________



Model: "discriminator"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         [(None, 2)]               0         
_________________________________________________________________
dense_3 (Dense)              (None, 100)               300       
_________________________________________________________________
dense_4 (Dense)              (None, 100)               10100     
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 101       
=================================================================
Total params: 10,501
Trainable params: 10,501
Non-trainable params: 0
_________________________________________________________________



The Simple_GAN model

The simple GAN model looks as follows :

class Simple_GAN(keras.Model) :
    
    def __init__(self, generator , discriminator , latent_generator, real_examples, 
                 batch_size=200, disc_iter=2, learning_rate=.005):
        pass
    
    # generator of real samples of size batch_size
    def get_real_sample(self):
        pass
    
    # generator of fake samples of size batch_size
    def get_fake_sample(self, training=True):
        pass
             
    # overloading train_step()
    def train_step(self,useless_argument): 
        pass


The goal of the exercise is to fill in the train_step method. But before we do so, let us fill in the other methods.




The __init__ method, the loss functions and the optimizers

Here is the code for the __init__ method:

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
       
The super.__init__() instruction instanciates some attributes of the model like for example self.history .

The loss of the generator is the mean squared error between discriminator(generator(latent_sample)) and the (generator) objective value 
.

The loss of the discriminator is the mean squared error between discriminator(fake_sample) and the (discriminator) objective value 
, summed with the mean squared error between discriminator(real_sample) and the (discriminator) objective value 
.

The optimizers are standard Adam optimizers.


The get_X_sample methods

A fake sample is just the image of the generator applied to a latent sample :

     def get_fake_sample(self, training=False):
        self.generator(self.latent_generator(self.batch_size), training=training)


A real sample is a random subset of the set of real_examples :

    def get_real_sample(self):
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices  = tf.random.shuffle(sorted_indices)[:self.batch_size]
        return tf.gather(self.real_examples, random_indices)



The train_step method (your shot)

Recall from the lesson on Keras models that to compute a gradient relatively to some variables the scheme is as follows :

x = tf.constant([.1, .2, .3, .4])                # x is a tensor 
with tf.GradientTape() as g:
  g.watch( x )                                   # we want to compute the gradient w/r to x
  y = f(x)                                       # y is 1-dimensional, f is a tensorflow function
gradient = g.gradient(y, x)                      # gradient is the gradient of f as at x
For example, if we want to train a model M to minimize a function f the scheme for one step looks like

with tf.GradientTape() as g:
  g.watch( M.trainable_variables )                
  y = f(M)                                        # y is 1-dimensional, f is a tensorflow function
gradient = g.gradient(y, M.trainable_variables)   # get the gradient of f at M
M.optimizer.apply_gradients(zip(gradient, M.trainable_variables))
Now one training step of our GANs consists in applying discr_iter times the gradient descent for the discriminator and then once for the generator.

Thus you are asked to fill in this method:

    def train_step(self,useless_argument): 
        pass
        #for _ in range(self.disc_iter) :
            
            # compute the loss for the discriminator in a tape watching the discriminator's weights
                # get a real sample
                # get a fake sample
                # compute the loss discr_loss of the discriminator on real and fake samples
            # apply gradient descent once to the discriminator

        # compute the loss for the generator in a tape watching the generator's weights 
            # get a fake sample 
            # compute the loss gen_loss of the generator on this sample
        # apply gradient descent to the discriminator
        
        # return {"discr_loss": discr_loss, "gen_loss": gen_loss}
Finally the whole class declaration (with the hole you have to fill in) of the class is :

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class Simple_GAN(keras.Model) :
    
    def __init__( self, generator , discriminator , latent_generator, real_examples, batch_size=200, disc_iter=2, learning_rate=.005):
        super().__init__()                         # run the __init__ of keras.Model first. 
        self.latent_generator = latent_generator
        self.real_examples    = real_examples
        self.generator        = generator
        self.discriminator    = discriminator
        self.batch_size       = batch_size
        self.disc_iter        = disc_iter
        
        self.learning_rate    = learning_rate
        self.beta_1=.5                               # standard value, but can be changed if necessary
        self.beta_2=.9                               # standard value, but can be changed if necessary
        
        # define the generator loss and optimizer:
        self.generator.loss      = lambda x : tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape))
        self.generator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.generator.compile(optimizer=generator.optimizer , loss=generator.loss )
        
        # define the discriminator loss and optimizer:
        self.discriminator.loss      = lambda x,y : tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape)) + tf.keras.losses.MeanSquaredError()(y, -1*tf.ones(y.shape))
        self.discriminator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.discriminator.compile(optimizer=discriminator.optimizer , loss=discriminator.loss )
       
    
    # generator of real samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        if not size :
            size= self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        if not size :
            size= self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices  = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)
             
    # overloading train_step()    
    def train_step(self,useless_argument):
        pass
        #for _ in range(self.disc_iter) :
            
            # compute the loss for the discriminator in a tape watching the discriminator's weights
                # get a real sample
                # get a fake sample
                # compute the loss discr_loss of the discriminator on real and fake samples
            # apply gradient descent once to the discriminator

        # compute the loss for the generator in a tape watching the generator's weights 
            # get a fake sample 
            # compute the loss gen_loss of the generator on this sample
        # apply gradient descent to the discriminator
        
        # return {"discr_loss": discr_loss, "gen_loss": gen_loss}