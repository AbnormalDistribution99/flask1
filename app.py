from crypt import methods
from distutils.log import debug
from flask import Flask, jsonify, request
import pickle
import numpy as np
import os
from flask_cors import CORS

filename = 'Diabetes_Prediction.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    
    data = {
            "message" : 'Hello user, thank you for pinging me. Get request succeedeed.', 
        }
  
    return jsonify(data)
	

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form.get('pregnancies'))
        glucose = int(request.form.get('glucose'))
        bp = int(request.form.get('bloodpressure'))
        st = int(request.form.get('skinthickness'))
        insulin = int(request.form.get('insulin'))
        bmi = float(request.form.get('bmi'))
        dpf = float(request.form.get('dpf'))
        age = int(request.form.get('age'))
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)
        
        
        my_prediction= my_prediction.tolist()
        data = {
            "message": "Thank you for your input",
            "result": my_prediction[0]
        }
        return jsonify(data)
    
if __name__ == '__main__':
	app.run(debug= True)