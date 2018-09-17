from keras.models import model_from_json
import numpy as np
from keras.preprocessing.image import img_to_array, load_img


def load_model(model_path, weights_path):

    # global model

    with open(model_path, 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights(weights_path)
    return model


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target)
    image = img_to_array(image)/255.
    image = np.expand_dims(image, axis=0)
    # return the processed image
    return image


def make_predictions(image, model):
    image = prepare_image(image, (250, 250))
    prediction = model.predict(image)
    return prediction


def decode_prediction(prediction, classes_path):
    with open(classes_path, 'r') as f:
        classes = f.read().splitlines()
    my_zip = zip(classes, prediction[0])
    sorted_tuples = sorted(my_zip, reverse=True, key=lambda pair: pair[1])[0:5]
    result_dict = {}
    for element in sorted_tuples:
        result_dict[element[0]] = element[1]
    return result_dict


if __name__ == "__main__":
    image = load_img('images/7669094_l.jpg')
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    model_path = 'dogbreeds_model_architecture.json'
    weights_path = 'dogbreeds_model_weights.h5'
    model = load_model(model_path, weights_path)
    # print(model.summary())

    image = load_img('images/shepherd.jpg')
    preds = make_predictions(image, model)
    preds_dec = decode_prediction(preds, 'classes.txt')
    with open('classes.txt', 'r') as f:
        classes = f.read().splitlines()
    print(preds_dec)
    # app.run()
