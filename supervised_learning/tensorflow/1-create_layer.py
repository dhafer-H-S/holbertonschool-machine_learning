import tensorflow as tf

"""create layer """
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
        """ a function tthat creat layers based on the inpute features"""
        """VarianceScaling initializer is used to initialize 
        weights in a neural network layer
        and here it's configured to use the fan_avg """
        initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
        """
        tf.layers.dense function is used to create such dense
        (fully connected) layers in TensorFlow
        """
        """specifies the number of neurons in the layer"""
        """sets the initializer for the layer's weights to the initializer"""
        layer = tf.layers.Dense(units=n, activation=activation, kernel_initializer=initializer, name="layer")
        """
        returns the output tensor of the layer, *
        which will be the result of applying the activation function
        to the weighted sum of inputs from the previous layer.
         """
        output = layer(prev)

        return output
