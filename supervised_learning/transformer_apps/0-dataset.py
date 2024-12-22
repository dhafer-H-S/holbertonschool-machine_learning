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
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train',
                                    as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation',
                                    as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        creates sub-word tokenizers for our dataset
        Args:
            data: tf.data.Dataset whose examples are formatted as a tuple (pt, en)
                pt: tf.Tensor containing the Portuguese sentence
                en: tf.Tensor containing the corresponding English sentence
        Returns: tokenizer_pt, tokenizer_en
            tokenizer_pt is the Portuguese tokenizer
            tokenizer_en is the English tokenizer
        """
        tokenizers = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )
        tokenizer_pt = tokenizers
        tokenizer_en = tokenizers
        return tokenizer_pt, tokenizer_en
