from flask import Flask, request, jsonify, render_template
import xgboost as xgb
import numpy as np
import pickle

app = Flask(__name__)

# Load the saved XGBoost model
with open("best_xgboost_model (2).pkl", "rb") as file:
    model = pickle.load(file)

# Load the saved scaler
with open("scaler (3).pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')  # Ensure this file exists

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form or JSON
        if request.form:
            input_data = [float(request.form.get(feat)) for feat in request.form]
        else:
            input_data = request.get_json(force=True)
            input_data = [input_data[feat] for feat in input_data]

        # Convert to array and reshape
        data = np.array(input_data).reshape(1, -1)

        # Normalize input data using the scaler
        normalized_data = scaler.transform(data)

        # Predict using the model
        prediction = model.predict(normalized_data)
        result = int(prediction[0])

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
