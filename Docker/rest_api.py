# import the necessary packages
from predict import load_model, make_predictions, decode_prediction
from PIL import Image
import flask
import io
import pandas as pd

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    results = {"success": False}

    model_path = 'dogbreeds_model_architecture.json'
    weights_path = 'dogbreeds_model_weights.h5'
    class_path = 'classes.txt'
    model = load_model(model_path, weights_path)

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = make_predictions(image, model)
            preds_dec = decode_prediction(preds, class_path)
            # indicate that the request was a success
            predictions = pd.Series(preds_dec).sort_values(ascending=False)
            results["predictions"] = predictions.to_json()
            results["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(results)
    # return results


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    app.run(debug=True, host='0.0.0.0', port=5000)
