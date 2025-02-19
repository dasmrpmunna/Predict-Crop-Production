import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__, static_folder="static")

#lode pickle model/file
flask_app = Flask(__name__)
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


#create home route
@flask_app.route("/")
def Home():
    return render_template("index.html")


#create predict route
@flask_app.route("/predict", methods=["POST"])
def predict():
    float_feature = [float(x) for x in request.form.values()]
    features= [np.array(float_feature)]
    prediction = model.predict(features)
    return render_template("index.html", Prediction_text="The Predicted Crops is: {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug = True)