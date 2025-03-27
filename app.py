#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load Trained model
model = joblib.load("bike_rental_model.pkl")

# Define input feature names (modify based on your dataset)
feature_names = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
                 "weathersit", "temp", "atemp", "hum", "windspeed"]

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from form
        features = [float(request.form[feature]) for feature in feature_names]
    except ValueError:
        return render_template("result.html", prediction="Invalid input. Please enter numeric values.")

    # Make prediction
    prediction = model.predict([features])[0]

    return render_template("result.html", prediction=f"Predicted bike rentals: {int(prediction)}")

if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




