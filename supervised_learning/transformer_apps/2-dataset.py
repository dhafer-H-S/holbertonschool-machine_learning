#!/usr/bin/env python3

"""
This task is about creating a class that loads and preps
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
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train',
                                    as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation',
                                    as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for our dataset
        Args:
            data: tf.data.Dataset whose examples are formatted as a
            tuple (pt, en)
                pt: tf.Tensor containing the Portuguese sentence
                en: tf.Tensor containing the corresponding English sentence
        Returns: tokenizer_pt, tokenizer_en
            tokenizer_pt is the Portuguese tokenizer
            tokenizer_en is the English tokenizer
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased")
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased")

        def iterate_pt():
            """Generate Portuguese sentences one at a time from the dataset"""
            for pt, _ in data:
                yield pt.numpy().decode('utf-8').strip()

        def iterate_en():
            """Generate English sentences one at a time from the dataset"""
            for _, en in data:
                yield en.numpy().decode('utf-8').strip()

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            iterate_pt(), vocab_size=2**13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            iterate_en(), vocab_size=2**13)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens
        Args:
            pt: tf.Tensor containing the Portuguese sentence
            en: tf.Tensor containing the corresponding English sentence
        Returns: pt_tokens, en_tokens
            pt_tokens is a tf.Tensor containing the Portuguese tokens
            en_tokens is a tf.Tensor containing the English tokens
        """
        pt_tokens = [self.tokenizer_pt.vocab_size] + \
            self.tokenizer_pt.encode(pt.numpy().decode('utf-8'),
                                     add_special_tokens=False) + \
            [self.tokenizer_pt.vocab_size + 1]
        en_tokens = [self.tokenizer_en.vocab_size] + \
            self.tokenizer_en.encode(en.numpy().decode('utf-8'),
                                     add_special_tokens=False) + \
            [self.tokenizer_en.vocab_size + 1]
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Acts as a TensorFlow wrapper for the encode instance method
        Args:
            pt: tf.Tensor containing the Portuguese sentence
            en: tf.Tensor containing the corresponding English sentence
        Returns: pt, en
            pt is a tf.Tensor containing the Portuguese tokens
            en is a tf.Tensor containing the English tokens
        """
        pt, en = tf.py_function(func=self.encode,
                                inp=[pt, en],
                                Tout=[tf.int64, tf.int64])
        pt.set_shape([None])
        en.set_shape([None])
        return pt, en
