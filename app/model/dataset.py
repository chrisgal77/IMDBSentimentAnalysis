import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import pandas as pd 


BUFFER_SIZE = 50000


def labeler(example, label):
    return example, tf.cast(label, tf.int64)


def get_imdb_dataset(root: str):
    ds_train_neg = tf.data.TextLineDataset(os.path.join(root, 'train', 'neg.csv')).skip(1)
    ds_train_neg = ds_train_neg.map(lambda x: labeler(x, 1))
    
    ds_train_pos = tf.data.TextLineDataset(os.path.join(root, 'train', 'pos.csv')).skip(1)
    ds_train_pos = ds_train_pos.map(lambda x: labeler(x, 0))

    ds_train = ds_train_neg.concatenate(ds_train_pos)

    ds_test_neg = tf.data.TextLineDataset(os.path.join(root, 'test', 'neg.csv')).skip(1)
    ds_test_neg = ds_test_neg.map(lambda x: labeler(x, 1))
    
    ds_test_pos = tf.data.TextLineDataset(os.path.join(root, 'test', 'pos.csv')).skip(1)
    ds_test_pos = ds_test_pos.map(lambda x: labeler(x, 0))
    
    ds_test = ds_test_neg.concatenate(ds_test_pos)
    
    ds_train = ds_train.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
    ds_test = ds_test.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
           
    return ds_train_neg, ds_train_pos
