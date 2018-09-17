from keras.callbacks import History, TensorBoard, CSVLogger
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras import applications
from keras import Model
from keras import optimizers
from keras import metrics
import functools
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
import utils as ut
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import keras
from sklearn.model_selection import GridSearchCV


def create_model(**kwargs):

    # verbose
    verbose = kwargs.pop('verbose', False)

    # input shape
    img_width = kwargs.pop('img_width', 250)
    img_height = kwargs.pop('img_height', 250)

    if K.image_data_format() == 'channels_first':
            input_shape = (3, img_width, img_height)
    else:
            input_shape = (img_width, img_height, 3)

    # base model class + frozen layers
    base_model_class = kwargs.pop('base_model_class', None)
    tune_layers = kwargs.pop('tune_layers', None)

    if base_model_class:
        base_model = base_model_class(include_top=False,
                                      input_shape=input_shape)
        if tune_layers:
            for layer in base_model.layers[:-tune_layers]:
                layer.trainable = False
        # print(base_model.summary())

    # classes (binary or multclass)
    no_classes = kwargs.pop('no_classes', 120)
    if no_classes == 2:
        final_activation = 'sigmoid'
        no_classes = 1  # binary classficication
    else:
        final_activation = 'softmax'

    # model
    if base_model_class:
        top_model = Sequential()
        top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        top_model.add(Dense(256))
        top_model.add(Activation('relu'))
        top_model.add(Dropout(0.3))

        # final layer
        top_model.add(Dense(no_classes))
        top_model.add(Activation(final_activation))

        model = Model(inputs=base_model.input,
                      outputs=top_model(base_model.output))

    else:
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # final layer
        model.add(Dense(no_classes))
        model.add(Activation(final_activation))

    # list of metrics
    metric_list = kwargs.pop('metric_list', [])

    # optimizer
    optimizer_class = kwargs.pop('optimizer_class', optimizers.Adam)
    optimizer_kwargs = kwargs.pop('optimizer_kwargs', {})
    optimizer = optimizer_class(**optimizer_kwargs)

    # loss
    if no_classes <= 2:
        default_loss = 'binary_crossentropy'
    else:
        default_loss = 'categorical_crossentropy'
    loss = kwargs.pop('loss', default_loss)

    # compile model
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metric_list)

    if verbose:
        # not so nice
        print('base_model is {}'.format(base_model.summary()))
        print('top_model is {}'.format(top_model.summary()))
        print('complete model is {}'.format(model.summary()))

    return model


class KerasConvNet(BaseEstimator, TransformerMixin):

    def __init__(self, batch_size,
                 epochs, image_gen_train,
                 image_dir,
                 optimizer=optimizers.Adam,
                 lr=None,
                 tune_layers=0,
                 base_model_class=None,
                 metric_list=[],
                 **kwargs):
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.image_gen_train = image_gen_train
        self.image_gen_test = None
        self.image_dir = image_dir
        self.target_size = (250, 250)

        # model params
        self.base_model_class = base_model_class
        self.tune_layers = tune_layers
        self.optimizer = optimizer
        self.lr = lr
        self.optimizer_kwargs = dict(lr=self.lr)
        self.metric_list = metric_list

    def fit(self, X, y):
        nb_train_samples = len(X)
        # base_model_class = applications.ResNet50
        base_model_class = self.base_model_class

        # top20_acc = functools.partial(metrics.top_k_categorical_accuracy, k=20)
        # top20_acc.__name__ = 'top20_acc'

        # metric_list = [metrics.categorical_accuracy, top20_acc]
        metric_list = self.metric_list
        # optimizer = optimizers.Adam
        optimizer = self.optimizer
        # optimizer_kwargs = dict(epsilon=1e-5,
        #                         lr=0.0001)
        optimizer_kwargs = self.optimizer_kwargs

        configs = dict(img_width=250,
                       base_model_class=base_model_class,
                       tune_layers=4,
                       optimizer=optimizer,
                       optimizer_kwargs=optimizer_kwargs,
                       verbose=False,
                       metric_list=metric_list)

        self.model = create_model(**configs)
        # callbacks

        train_generator = ut.CustomImageGenerator(self.image_dir,
                                                  X,
                                                  y,
                                                  self.image_gen_train,
                                                  self.batch_size,
                                                  self.target_size)

        with tf.device('/gpu:0'):
            self.model.fit_generator(
                train_generator.get_next_batch(),
                steps_per_epoch=nb_train_samples // self.batch_size,
                epochs=self.epochs)

        return self

    def predict(self, X):
        image_gen_test = ImageDataGenerator(
            rescale=1. / 255)
        # maybe better to inherit from keras.utils.Sequence in future
        test_generator = ut.CustomImageGenerator(self.image_dir,
                                                 X,
                                                 None,
                                                 image_gen=image_gen_test,
                                                 batch_size=self.batch_size,
                                                 target_size=self.target_size)
        steps = len(X) // self.batch_size
        preds = self.model.predict_generator(test_generator.get_next_batch(),
                                             steps=steps)
        return preds

    def score(self, X_test, y_test):
        image_gen_test = ImageDataGenerator(
            rescale=1. / 255)
        # maybe better to inherit from keras.utils.Sequence in future
        test_generator = ut.CustomImageGenerator(self.image_dir,
                                                 X_test,
                                                 y_test,
                                                 image_gen=image_gen_test,
                                                 batch_size=self.batch_size,
                                                 target_size=self.target_size)
        steps = len(X_test) // self.batch_size
        #print(self.model.metrics_names)
        eval = self.model.evaluate_generator(test_generator.get_next_batch(),
                                             steps=steps)[0]
        # if K.backend() == 'tensorflow':
        #         K.clear_session()
        return eval


if __name__ == '__main__':
    base_model_class = applications.inception_resnet_v2.InceptionResNetV2
    # base_model_class = applications.ResNet50

    top20_acc = functools.partial(metrics.top_k_categorical_accuracy, k=20)
    top20_acc.__name__ = 'top20_acc'

    metric_list = [metrics.categorical_accuracy, top20_acc]

    optimizer = optimizers.Adam
    optimizer_kwargs = dict(epsilon=1e-5,
                            lr=0.0001)

    configs = dict(img_width=200,
                   base_model_class=base_model_class,
                   tune_layers=4,
                   optimizer=optimizer,
                   optimizer_kwargs=optimizer_kwargs,
                   verbose=False)

    print(create_model(**configs))
    image_gen_train = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True)

    image_dir = "../data/train/"
    label_path = "../data/labels.csv"
    df = pd.read_csv(label_path)
    list_of_labelnames = np.array(ut.one_hot(df['breed'].values))
    list_of_images = np.array([id + '.jpg' for id in df['id'].values])
    X_train, y_train = list_of_images[:500], list_of_labelnames[:500]

    conv = KerasConvNet(32, 20, image_gen_train, image_dir,
                        optimizer, lr=0.001, tune_layers=2,
                        base_model_class=base_model_class,
                        metric_list=metric_list)

    #conv.fit(X_train, y_train)
    #preds = conv.predict(X_train)
    #print(preds.shape)
    #print(conv.score(X_train, y_train))
    param_grid = {'lr': [0.001, 0.01],
                  'batch_size': [32],
                  'epochs': [2],
                  'tune_layers': [4],
                  'optimizer': [optimizers.Adam, optimizers.RMSprop,
                                optimizers.SGD, optimizers.Nadam]}
    print(conv.get_params())
    grid = GridSearchCV(conv, param_grid, verbose=2, cv=3, n_jobs=1)
    grid.fit(X_train, y_train)
    df = pd.DataFrame(grid.cv_results_)
    df.to_pickle('cv.pkl')
    #print(create_model())
