#!/usr/bin/env python3

import os
import random
import numpy as np
import tensorflow as tf
Decoder = __import__('10-transformer_decoder').Decoder

SEED = 0

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

decoder = Decoder(N=6, dm=512, h=8, hidden=2048, target_vocab=8000, max_seq_len=10000, drop_rate=0.1)
print(decoder.dm)
print(decoder.N)
print(decoder.embedding)
print(decoder.positional_encoding)
print(decoder.blocks)
print(decoder.dropout)
x = tf.random.uniform((32, 15))
hidden_states = tf.random.uniform((32, 10, 512))
output = decoder(x, hidden_states, training=True, look_ahead_mask=None, padding_mask=None)
print(output)
