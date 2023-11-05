#!/usr/bin/env python3
"""Implement my own transfer learning on cifar 10."""
import tensorflow.keras as K
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical


def get_data():
    """Initialize cifar-10 data."""
    (X_train, Y_train), (X_valid, Y_valid) = K.datasets.cifar10.load_data()
    return (X_train, Y_train), (X_valid, Y_valid)


def preprocess_data(X, Y):
    """
    Preprocess data for ResNet-50 model.

    Parameters:
        X (numpy.ndarray): Input data (images).
        Y (numpy.ndarray): Labels.

    Returns:
        Tuple: Preprocessed input data and labels.
    """
    # Resize images to the required input size of ResNet-50 (224x224 pixels)
    X_resized = tf.image.resize(X, (32, 32))

    # Convert labels to one-hot encoding
    Y_one_hot = tf.one_hot(Y, 10)

    # Preprocess input data using ResNet-50 specific preprocessing function
    X_preprocessed = preprocess_input(X_resized)

    return X_preprocessed, Y_one_hot


def create_model():
    """Create model from ResNet50."""
    base_model = K.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(32, 32, 3),
        pooling=None)

    base_model.trainable = False

    x = base_model.output
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.5)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.Model(base_model.input, outputs)

    return model


def preprocess_data(X, Y):
    """
    Function to preprocess data for the model.
    X is a numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data.
    Y is a numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X.
    Returns: X_p, Y_p where
    X_p is a numpy.ndarray containing the preprocessed X.
    Y_p is a numpy.ndarray containing the preprocessed Y.
    """
    X_p = K.applications.resnet50.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == "__main__":
    # Load the CIFAR10 dataset
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    # Preprocess the data
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # Create a ResNet50 model with weights pre-trained on ImageNet
    base_model = K.applications.ResNet50(weights='imagenet',
                                         include_top=False,
                                         input_shape=(224, 224, 3))

    # Freeze the base model
    base_model.trainable = False

    # Create new model on top
    inputs = K.Input(shape=(32, 32, 3))
    x = K.layers.Lambda(lambda img: tf.image.resize(img, (224, 224)))(inputs)
    x = base_model(x, training=False)
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.5)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)
    model = K.Model(inputs, outputs)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, Y_train,
                        validation_data=(X_test, Y_test),
                        batch_size=32,
                        epochs=10,
                        verbose=1)

    # Save the model
    model.save('cifar10.h5')
