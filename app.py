import numpy as np
from flask import Flask, request, render_template
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained model from the pickle file
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]  # Convert input values to float
    features = [np.array(float_features)]
    prediction = model.predict(features)  # Make prediction

    # Return result to the HTML page
    return render_template("index.html", Prediction_text="The Predicted Crop is: {}".format(prediction[0]))

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
