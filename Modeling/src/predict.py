import pandas as pd
import numpy as np
import pickle
from keras.models import load_model
import functools
from keras.metrics import top_k_categorical_accuracy
import keras
import utils_early_stopping as ut
from keras.utils.generic_utils import get_custom_objects
import os

# load model (need to add the "custom" metric function again)
top20_acc = functools.partial(top_k_categorical_accuracy, k=20)
top20_acc.__name__ = 'top20_acc'
keras.losses.custom_loss = top20_acc
get_custom_objects().update({"top20_acc": top20_acc})

print("check of loading model works...")
model = load_model('my_model_inception_2_full.h5')
print(model.summary())

test_path = "../data/test"
image_list = os.listdir(test_path)

gen = ut.predictGenerator(test_path, add_names=True)


def make_predictions(generator, model):
    df = pd.DataFrame()
    preds = []
    ids = []
    i = 0
    for batch in generator:
        preds= model.predict(batch[0])
        ids = batch[1]
        print(ids)
        print(preds)
        pred_temp = pd.DataFrame(preds)
        id_temp = pd.DataFrame({'ids': ids})
        temp = id_temp.join(pred_temp)
        df = df.append(temp)
        # i += 1
        # if i > 2:
        # break
    return df


df= make_predictions(gen, model)

# df = pd.DataFrame({'ids': ids, 'preds': preds})
df.to_csv('preds.csv')
print(df)




#print(image_list[:3])

#df_test = pd.read_pickle("test_df.pkl")
#print(df_test.head())
