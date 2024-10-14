#!/usr/bin/env python3

import tensorflow.keras as keras
autoencoder = __import__('2-convolutional').autoencoder

def main():
    # Define parameters
    input_dims = 784
    hidden_layers = [128, 64]
    latent_dims = 32
    lambtha = 10e-6

    # Create the autoencoder
    encoder, decoder, auto = autoencoder(input_dims, hidden_layers, latent_dims, lambtha)

    # Check if the models are built properly
    print(isinstance(encoder, keras.Model))
    print(isinstance(decoder, keras.Model))
    print(isinstance(auto, keras.Model))

if __name__ == "__main__":
    main()