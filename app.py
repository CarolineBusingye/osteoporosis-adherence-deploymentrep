from flask import Flask, request, jsonify, render_template
import xgboost as xgb
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the saved XGBoost model
with open("xgb_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')  # Create this HTML for web form inputs

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form or JSON
        if request.form:
            input_data = [float(request.form.get(feat)) for feat in request.form]
        else:
            input_data = request.get_json(force=True)
            input_data = [input_data[feat] for feat in input_data]

        # Convert to correct format
        data = np.array(input_data).reshape(1, -1)
        
        # Predict
        prediction = model.predict(data)
        result = int(prediction[0])  # Or float if probability

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
