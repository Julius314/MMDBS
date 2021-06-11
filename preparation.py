import tensorflow as tf
import numpy as np

def convertarray(a):

    # change type to float
    a = a.astype(np.float32)
    # range conversion
    a = np.interp(a, [0, 255], [0, 1])

    return a


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = convertarray(x_train)
x_test = convertarray(x_test)

def loadData():

    return x_train,y_train,x_test,y_test
    
