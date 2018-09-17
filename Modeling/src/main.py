#!/usr/bin/env python
import utils as ut
from keras.models import load_model
import pandas as pd
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
import pickle

print("Hello")

model = load_model('my_model.h5')
print(model.summary())

# path

image_dir = "/home/jodahr/PycharmProjects/DogBreeds/data/test/"
image_list = np.array(os.listdir(image_dir))
print(image_list)
image_gen_test = ImageDataGenerator(
    rescale=1. / 255)
ci = ut.CustomImageGenerator(image_dir, image_list, image_gen=image_gen_test)
print(next(ci.get_next_batch()))

#test = model.predict_generator(ci.get_next_batch(), steps = len(image_list)//128)
#print(test)

with open('label_to_id.pkl', 'rb') as f:
    label_to_id = pickle.load(f)

with open('id_to_label.pkl', 'rb') as f:
    id_to_label = pickle.load(f)

print(id_to_label)

gen = ut.predictGenerator(image_dir,target_size=(150,150), add_names=True)
#print(next(gen))
classes = list(id_to_label.keys())
df = ut.get_results(gen, model, classes)
df.sort_values('score', ascending=False).to_csv("predictions.csv")
