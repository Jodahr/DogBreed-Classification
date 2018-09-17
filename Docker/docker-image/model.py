#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Simple Keras Code for creating a Dog Breed Image Classifier """

# System packages
import os
import sys
import shutil
import subprocess

# Standard data science packages
import numpy as np

# Keras and tensorflow
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPool2D
from keras import metrics, Model
from keras import backend as K
from keras.callbacks import History, TensorBoard
from keras import applications
from keras.preprocessing.image import ImageDataGenerator,\
    array_to_img, img_to_array, load_img
import tensorflow as tf
import json


# data path
train_data_dir = '/mnt/DataDisk/jodahr/data/dog_breeds/training/'
validation_data_dir = '/mnt/DataDisk/jodahr/data/dog_breeds/validation/'

command = 'find ' + train_data_dir + ' -type l  | wc -l'
nb_train_samples = int(os.popen(command).read())
command = 'find ' + validation_data_dir + ' -type l  | wc -l'
nb_validation_samples = int(os.popen(command).read())

epochs = 30
batch_size = 64
img_width, img_height = 250, 250

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# this is the augmentation configuration for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.5,
    zoom_range=0.5,
    horizontal_flip=True)


# only rescaling for validation
test_datagen = ImageDataGenerator(rescale=1. / 255)

# train and validation generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

base_model = applications.ResNet50(include_top=False,
                                   input_shape=input_shape)

# Load VGG16 model
base_model = applications.VGG16(include_top=False,
input_shape=input_shape)

print(base_model.summary())

# keep the last conv layer traibable
for layer in base_model.layers[:-2]:
    layer.trainable = False

print(base_model.summary())

x = base_model.output

x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(units=512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(units=256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(units=120, activation='softmax')(x)

model = Model(input=base_model.input, output=x)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[metrics.categorical_accuracy])

# callbacks
history = History()
tb = TensorBoard()

with tf.device('/gpu:0'):
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[history, tb])

# Save the weights
model.save_weights('dogbreeds_model_weights.h5')

# Save the model architecture
with open('dogbreeds_model_architecture.json', 'w') as f:
    f.write(model.to_json())

# # Model reconstruction from JSON file
# with open('model_architecture.json', 'r') as f:
#     model = model_from_json(f.read())

# # Load weights into the new model
# model.load_weights('model_weights.h5')

print(K.tensorflow_backend._get_available_gpus())
