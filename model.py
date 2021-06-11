import pydot
import graphviz
import numpy as np
from preparation import loadData
from sanity import performSanityCheck
from tensorflow.keras.models import Model
from tensorflow.keras.layers import AveragePooling2D, Activation, Input, LeakyReLU, Conv2D, MaxPooling2D,UpSampling2D, Flatten, Dense, Reshape, BatchNormalization, Conv2DTranspose
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam

import time
#load data
x_train, y_train, x_test, y_test = loadData()

print("data successfully loaded")

#print shapes
print(f"x_train shape: {np.shape(x_train)}")
#print(f"y_train shape: {np.shape(y_train)}")
print(f"x_test shape: {np.shape(x_test)}")
#print(f"y_test shape: {np.shape(y_test)}")

print("start building model...")

#Model Architecture

#Input
img_shape = (32, 32, 3)
input = Input(shape=img_shape)#Input dim (image_height, image_width, color_channels)
latent_dim = 10

#Encoded layers

encode = Conv2D(8, (5, 5), activation='relu', padding='same')(input)
encode = AveragePooling2D((2, 2), padding='same')(encode)
encode = Conv2D(16, (5, 5), activation='relu', padding='same')(encode)
encode = AveragePooling2D((2, 2), padding='same')(encode)
encode = Conv2D(32, (5, 5), activation='relu', padding='same')(encode)
encode = AveragePooling2D((2, 2), padding='same')(encode)
encode = Conv2D(64, (3, 3), activation='relu', padding='same')(encode)
encode = Flatten()(encode)
encode = Dense(latent_dim, activation='softmax')(encode)  # latent space

encoded_input = Input(shape=(latent_dim,))

#Decoded layers

decode = Dense(64)(encoded_input)
decode = Reshape((4, 4, 4))(decode)
decode = Conv2D(64, (3, 3), activation='relu', padding='same')(decode)
decode = BatchNormalization()(decode)
decode = Conv2DTranspose(32, (5, 5), strides=2, activation='relu', padding='same')(decode)
decode = BatchNormalization()(decode)
decode = Conv2DTranspose(16, (5, 5), strides=2, activation='relu', padding='same')(decode)
decode = BatchNormalization()(decode)
decode = Conv2DTranspose(8, (5, 5), strides=2, activation='relu', padding='same')(decode)
decode = BatchNormalization()(decode)
decode = Conv2D(3, (5, 5), activation='sigmoid', padding='same')(decode)


#encoder model
encoder = Model(input, encode, name="Encoder")

#decoder model
decoder = Model(encoded_input, decode, name="Decoder")

#autoencoder model
autoencoder = Model(input, decoder(encoder(input)), name="Autoencoder")

#print summary of layers
encoder.summary()
decoder.summary()
autoencoder.summary()

#compile model 
autoencoder.compile(
    optimizer=Adam(lr=0.0007),
    loss='mse',
    metrics=['accuracy'])


idstring = time.strftime("%Y%m%d-%H%M%S")

#create tensorboard for visualization
NAME = "autoencoder-" + idstring
tensorboard = TensorBoard(log_dir='logs\{}'.format(NAME)) #for windows 

#Train model
autoencoder.fit(
    x_train,
    x_train,
    epochs=100,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test),
    callbacks=[tensorboard])

#save model
autoencoder.save(f'models/autoencoder-{idstring}.h5')
encoder.save(f'models/encoder-{idstring}.h5')
decoder.save(f'models/decoder-{idstring}.h5')

#command for running tensorboard graph in Terminal (using windows)
#tensorboard --logdir="path....."

#perform sanity check
performSanityCheck(autoencoder, encoder, x_test, y_test)


