import keras
import tensorflow as tf

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
generator, discriminator, latent_generator = fully_connected_GenDiscr([1,100,100,2], None)
print(generator.summary())
print(discriminator.summary())