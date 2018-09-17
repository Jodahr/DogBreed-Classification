#!/usr/bin/env python
import pandas as pd
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import applications
from keras import Model
from keras import metrics
import tensorflow as tf
import pickle
import keras
from sklearn.utils import class_weight
import functools


class CustomImageGenerator():
    """
    This is a custom class for an image data generator
    which uses an default ImageDataGenerator to produce batches of images.
    It needs directory where the images are located.
    Optionally, you can pass a list of image names
    and a list of one hot encoded labels.
    """
    def __init__(self, image_dir, image_filenames=None, label_list=None,
                 image_gen=ImageDataGenerator(),
                 batch_size=32, target_size=(250, 250)):
        """Constructor method.

        :param image_dir: directory which contains images
        :param image_filenames: list of image names
        :param label_list: list of one hot encoded labels (default None)
        :param image_gen: image generator used for flow method
        :param batch_size: number of images in a batch
        :param target_size: image target size
        :returns: None
        :rtype: None

        """
        self.image_gen = image_gen
        self.image_dir = image_dir
        self.image_filenames = image_filenames
        self.label_list = label_list
        self.batch_size = batch_size
        self.target_size = target_size

    def get_next_batch(self, shuffle=True):
        if self.image_filenames is None:
            self.image_filenames = os.listdir(self.image_dir)
        image_path_list = [self.image_dir + image_name
                           for image_name in self.image_filenames]
        if self.label_list is not None:
            samples = list(zip(image_path_list, self.label_list))
        else:
            samples = image_path_list
        if shuffle:
            np.random.shuffle(samples)
        chunks = [samples[x:x+self.batch_size]
                  for x in range(0, len(samples),
                                 self.batch_size)]
        while True:
            for chunk in chunks:
                if self.label_list is not None:
                    x, y = zip(*chunk)
                else:
                    x = chunk
                    y = None
                arr_batch = np.array([self.get_array(image)
                                    for image in x])
                yield self.image_gen.flow(arr_batch, y).next()
                #yield self.image_gen.flow(arr_batch, y)

    def get_array(self, image):
        img = load_img(image, target_size=self.target_size)
        img_arr = img_to_array(img)
        return img_arr


# not very elegant, but works
def predictGenerator(path, batch_size=32,
                     target_size=(250, 250), add_names=False):
    '''Generator wich returns batches of img arrays.
 Alternatively you can add the filenames.'''
    file_names = os.listdir(path)  # list of filenames
    chunks = [file_names[x:x+batch_size]
              for x in range(0, len(file_names),
                             batch_size)]  # divided into chunks

    for chunk in chunks:
        full_names = [path + '/' + file_name
                      for file_name in chunk]  # create full filenames
        img_batch = [load_img(full_name,
                              target_size=target_size)
                     for full_name in full_names]  # load image batches
        arr_batch = np.empty(shape=(1,
                                    target_size[0],
                                    target_size[1],
                                    3))  # empty multidimensional numpy array
        # rescale arrays and combine them
        for element in img_batch:
            img_arr = np.expand_dims(img_to_array(element)/255., axis=0)
            arr_batch = np.concatenate((arr_batch, img_arr), axis=0)
        # yield
        if add_names is False:
            yield arr_batch[1:, :, :, :]
        else:
            yield (arr_batch[1:, :, :, :], chunk, img_batch)


def get_results(my_generator_tuple, model, class_names):
    result = pd.DataFrame()
    for element in my_generator_tuple:
        predictions = model.predict(element[0])
        names = element[1]
        id = [name.split('.')[0] for name in names]
        images = element[2]
        temp = pd.DataFrame({'idx': predictions.argmax(axis=1),
                             'score': predictions.max(axis=1),
                             'filenames': names,
                             'img': images,
                             'id': id})
        temp['class_name'] = temp['idx'].apply(lambda x: class_names[x])
        result = pd.concat([result, temp])
    return result


def get_label_dict(array_of_labelnames):
    class_weights = class_weight.compute_class_weight(
        'balanced', np.unique(array_of_labelnames), array_of_labelnames)
    print(class_weights)
    unique = np.unique(array_of_labelnames)
    id_to_label = dict(enumerate(unique))
    label_to_id = {value: key for key, value in id_to_label.items()}
    return id_to_label, label_to_id, class_weights


def one_hot(array_of_labelnames, save_dict=True):
    id_to_label, label_to_id, _ = get_label_dict(array_of_labelnames)
    with open('label_to_id.pkl', 'wb') as f:
        pickle.dump(id_to_label, f)
    with open('id_to_label.pkl', 'wb') as f:
        pickle.dump(label_to_id, f)
    return [to_categorical(label_to_id[name],
                           len(label_to_id)) for name in array_of_labelnames]


def model():
    img_width, img_height = 250, 250
    if K.image_data_format() == 'channels_first':
            input_shape = (3, img_width, img_height)
    else:
            input_shape = (img_width, img_height, 3)

    # maybe pop some layers with model.layers.pop() ?
    base_model = applications.VGG16(include_top=False, input_shape=input_shape)
    #base_model = applications.ResNet50(include_top=False, input_shape=input_shape)
    for layer in base_model.layers[:-2]:
            layer.trainable = False
    # model topology
    top_model = Sequential()

    # top_model.add(Conv2D(128, (2, 2), input_shape=base_model.output_shape[1:]))
    # top_model.add(Activation('relu'))
    # top_model.add(MaxPooling2D(pool_size=(2, 2)))
    # top_model.add(Dropout(0.5))
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256))
    top_model.add(Activation('relu'))
    top_model.add(Dropout(0.2))
    top_model.add(Dense(120))
    top_model.add(Activation('softmax'))

    model = Model(inputs=base_model.input,
                  outputs=top_model(base_model.output))
    print(model.summary())
    # # functional method

    # model = Sequential()
    # model.add(Conv2D(128, (3, 3), input_shape=input_shape))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Conv2D(128, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Conv2D(256, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    # model.add(Dense(512))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(120))
    # model.add(Activation('softmax'))

    # model.compile(loss='binary_crossentropy',
    #                             optimizer='rmsprop',
    #                             metrics=['accuracy'])

    # x0 = Flatten()(base_model.output)
    # x1 = Dropout(0.2)(x0)
    # x2 = Dense(units=512, activation='relu')(x1)
    # x3 = Dropout(0.5)(x2)
    # x4 = Dense(units=120, activation='softmax')(x3)

        # # functional method
    # x0 = Conv2D(128,(2,2),activation='relu', input_shape=base_model.output_shape[1:])\
    #                                   (base_model.output)
    # x1 = MaxPooling2D(pool_size=(2,2))(x0)
    # x2 = Flatten()(x1)
    # #x3 = Dropout(0.2)(x2)
    # x4 = Dense(units=128, activation='relu')(x2)
    # x5 = Dropout(0.5)(x4)
    # x6 = Dense(units=120, activation='softmax')(x5)

    # # pls update to keras 2 api
    # model = Model(input=base_model.input, output=x4)
    # #model.summary()

    top20_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=20)
    top20_acc.__name__ = 'top20_acc'
    
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=[metrics.categorical_accuracy, top20_acc])

    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
    #               metrics=[metrics.categorical_accuracy])
    return model



if __name__ == '__main__':

    image_dir = "../data/train/"
    label_path = "../data/labels.csv"

    df = pd.read_csv(label_path)
    list_of_labelnames = np.array(one_hot(df['breed'].values))
    list_of_images = np.array([id + '.jpg' for id in df['id'].values])
    print(get_label_dict(df['breed'].values))
    _, _, class_weights = get_label_dict(df['breed'].values)
    #exit()
    
    print("total number of images are {}".format(len(list_of_labelnames)))
    # data_df = pd.DataFrame({'images': list_of_images,
    #                         'labels': list_of_labelnames})
    # print(data_df.head().to_dict())

    from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
    
    X_train, X_test, y_train, y_test = train_test_split(list_of_images,
                                                        list_of_labelnames,
                                                        test_size=.2,
                                                        random_state=42,
                                                        )
    sss = StratifiedShuffleSplit(n_splits=1, random_state=42, test_size=0.2)

    #print(next(sss.split(list_of_images,list_of_labelnames)))
    
    train_index, test_index = list(sss.split(list_of_images,
                                             list_of_labelnames))[0]
    X_train, y_train = list_of_images[train_index], list_of_labelnames[train_index]
    X_test, y_test = list_of_images[test_index], list_of_labelnames[test_index]

    print("total number of train images are {}".format(len(train_index)))
    print("total number of test images are {}".format(len(test_index)))
    
    image_gen_train = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True)

    image_gen_test = ImageDataGenerator(
        rescale=1. / 255)
    
    ci = CustomImageGenerator(image_dir, X_train, y_train,
                                                         image_gen=image_gen_train)
    #ci = CustomImageGenerator(image_dir, list_of_images)

    #print(ci.label_list)

#    batch = next(ci.get_next_batch())
 #   print(batch)

    nb_train_samples = len(X_train)
    nb_validation_samples = len(X_test)
    epochs = 40
    batch_size = 256

    print("\n\n\n")
    print(nb_train_samples)
    print(nb_validation_samples)
    print(nb_validation_samples // batch_size)
    
    model = model()
#    print(model.summary())

    #print(list_of_images)

    train_generator = CustomImageGenerator(image_dir,
                                           X_train,
                                           y_train,
                                           image_gen=image_gen_train,
                                           batch_size=batch_size)
    validation_generator = CustomImageGenerator(image_dir,
                                                X_test,
                                                y_test,
                                                image_gen=image_gen_test,
                                                batch_size=batch_size)

#    print(class_weights)
    
    with tf.device('/gpu:0'):
        model.fit_generator(
            train_generator.get_next_batch(),
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator.get_next_batch(),
            validation_steps=nb_validation_samples // batch_size,
            # max_queue_size=5
            #        callbacks = [history, tb]
            class_weight=class_weights)

    model.save('my_model_new2.h5')
