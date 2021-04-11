#Importing necessary packages
import numpy as np
from flask import Flask, request, render_template
import pickle
from fastai.tabular import *
import os

#Saving the working directory and model directory
cwd = os.getcwd()
path = cwd + '/model'

#Initializing the FLASK API
app = Flask(__name__)

#Loading the saved model using fastai's load_learner method
model = load_learner(path, 'model.pkl')

#Defining the home page for the web service
@app.route('/')
def home():
    return render_template('index.html')

#Writing api for inference using the loaded model
@app.route('/predict',methods=['POST'])

#Defining the predict method get input from the html page and to predict using the trained model

def predict():
    
   
    #all the input labels . We had only trained the model using these selected features.
    labels = ['year', 'mpg', 'fuelType', 'tax', 'transmission', 'engineSize', 'mileage']
        
    #Collecting values from the html form and converting into respective types as expected by the model
        
    year = float(request.form["year"])
    mpg =  float(request.form["mpg"])
    fuelType = request.form["fuelType"]
    tax = float(request.form["tax"])
    transmission = request.form['transmission']
    engineSize =  float(request.form["engineSize"])
    mileage = float(request.form["mileage"])

    #making a list of the collected features
    features = [ year, mpg, fuelType, tax, transmission, engineSize, mileage]

    #fastai predicts from a pandas series. so converting the list to a series
    to_predict = pd.Series(features, index = labels)
    
    #Getting the prediction from the model and rounding the float into 2 decimal places
    prediction = round(float(model.predict(to_predict)[1]),2)

    # Making all predictions below 0 dollers and above 20000 dollers as invalid
    if prediction > 0 and prediction <= 20000:
        return render_template('index.html', prediction_text='Your Input : {} Resale Cost: {}  Dollers'.format(features,prediction))
    else:
        return render_template('index.html', prediction_text='Invalid Prediction !! Network Unable To Predict For The Given Inputs')

    
if __name__ == "__main__":
    app.run(debug=True)