# -*- coding: utf-8 -*-
"""
Created on Sat Aug 1 2020
@author: Sai Keerthi
"""
import pickle
import pandas as pd
from flask import Flask, request
from flask_cors import CORS,cross_origin
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

pickle_in = open("scaler.pkl", "rb")
scaler = pickle.load(pickle_in)

pickle_in = open("model.pkl", "rb")
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome to Diabetes Predictor"

@app.route('/predict', methods=["GET"])
@cross_origin()
def predict():
    
    """Let's predict the Diabetes for the patient.
       Single Entity
    ---
    parameters:  
      - name: Pregnancies
        in: query
        type: number
        required: true
      - name: Glucose
        in: query
        type: number
        required: true
      - name: Blood Pressure
        in: query
        type: number
        required: true
      - name: Skin Thickness
        in: query
        type: number
        required: true
      - name: Insulin
        in: query
        type: number
        required: true
      - name: BMI
        in: query
        type: number
        required: true
      - name: Diabetes Pedigree Function
        in: query
        type: number
        required: true
      - name: Age
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """

    Pregnancies              = request.args.get("Pregnancies")
    Glucose                  = request.args.get("Glucose")
    BloodPressure            = request.args.get("Blood Pressure")
    SkinThickness            = request.args.get("Skin Thickness")
    Insulin                  = request.args.get("Insulin")
    BMI                      = request.args.get("BMI")
    DiabetesPedigreeFunction = request.args.get("Diabetes Pedigree Function")
    Age                      = request.args.get("Age")

    prediction = classifier.predict(scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]))
    result = prediction[0]
    print(result)

    if result == 1:
        text_result = 'DIABETIC'
    else:
        text_result = 'NON-DIABETIC'

    return "Hello, The prediction for the given patient's inputs is " + text_result

@app.route('/predict_file', methods=["POST"])
@cross_origin()
def predict_file():
    """Let's predict the Diabetes in Bulk - Input the File
       Bulk Predict from a CSV File
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output
        
    """

    df_file = pd.read_csv(request.files.get("file"))
    print(df_file.head())
    prediction = classifier.predict(scaler.transform(df_file))
    result = list(prediction)

    text_result = []
    for i in result:
        if i == 1:
            text_result.append('DIABETIC')
        else:
            text_result.append('NON-DIABETIC')

    return "Hello, The prediction for the given patients in input file is " + str(text_result)

if __name__=='__main__':
    #to run on cloud
	app.run(debug=True) # running the app

    #to run locally
    #app.run(host='127.0.0.1', port=8000, debug=True)
