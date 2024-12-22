#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_datasets as tfds

pt2en_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
for pt, en in pt2en_train.take(1):
    pt_text = pt.numpy().decode('utf-8').strip()
    en_text = en.numpy().decode('utf-8').strip()
    print(pt_text)
    print(en_text)