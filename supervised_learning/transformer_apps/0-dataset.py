#!/usr/bin/env python3

"""
this task is about creating a class that loads and preps
a dataset for machine translation
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import transformers

class Dataset:
    """
    This class loads and preps a dataset for machine translation
    """
    def __init__(self):
        """
        Class constructor
        """
        self.data_train = self.load_and_strip('train')
        self.data_valid = self.load_and_strip('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def load_and_strip(self, split):
        """
        Loads the dataset and strips whitespace from the text
        Args:
            split: the split of the dataset to load (train or validation)
        Returns:
            A tf.data.Dataset with stripped text
        """
        data = tfds.load('ted_hrlr_translate/pt_to_en', split=split, as_supervised=True)
        return data.map(lambda pt, en: (tf.strings.strip(pt), tf.strings.strip(en)))

    def tokenize_dataset(self, data):
        """
        creates sub-word tokenizers for our dataset
        Args:
            data: tf.data.Dataset whose examples are formatted as a
        """
        # Tokenizer creation logic here
        # For example:
        tokenizer_pt = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        tokenizer_en = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        return tokenizer_pt, tokenizer_en
