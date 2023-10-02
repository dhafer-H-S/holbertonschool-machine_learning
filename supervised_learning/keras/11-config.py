#!/usr/bin/env python3ù
"""save model configuration in JSON and load model configuration from JSON"""


def save_config(network, filename):
    """network is the model whose weights should be saved
    filename is the path of the file that the weights should be saved to"""
    json_config = network.to_json()
    with open(filename, 'w') as json_file:
        json_file.write(json_config)


def load_config(filename):
    """filename is the path of the file containing
    the model’s configuration in JSON format"""
    with open(filename, 'r') as json_file:
        json_config = json_file.read()
    return K.models.model_from_json(json_config)