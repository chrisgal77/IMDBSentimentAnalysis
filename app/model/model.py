import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import pandas as pd


def get_vocabulary(dataset):
    pass


def get_imdb_dataset(root: str):
    ds_train = tf.data.TextLineDataset(os.path.join(root, 'train.csv'))
    ds_test = tf.data.TextLineDataset(os.path.join(root, 'test.csv'))
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    
    return ds_train, ds_test


if __name__ == "__main__":
    
    ds_train, ds_test = get_imdb_dataset('/home/gal/Documents/Projects/IMDBSentimentAnalysis/dataset')
    print(ds_train)