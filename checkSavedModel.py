import tensorflow as tf
from preparation import loadData
from sanity import performSanityCheck
import numpy as np

#load data
x_train, y_train, x_test, y_test = loadData()


encoder = tf.keras.models.load_model('models/encoder-final.h5')
autoencoder = tf.keras.models.load_model('models/autoencoder-final.h5')

#perform sanity check
performSanityCheck(autoencoder, encoder, x_test, y_test)
