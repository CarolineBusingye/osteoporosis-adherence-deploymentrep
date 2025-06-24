from flask import Flask, request, jsonify, render_template
import xgboost as xgb
import numpy as np
import joblib


app = Flask(__name__)
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Manually reconstruct MinMaxScaler
scaler = MinMaxScaler()
scaler.min_ = np.array([0., 0., 0.])  # calculated as -min/(max-min)
scaler.scale_ = 1 / (np.array([90.  1.  1.]) - np.array([18.  0.  0..]))
scaler.data_min_ = np.array([18.  0.  0.])
scaler.data_max_ = np.array([90.  1.  1.])
scaler.data_range_ = scaler.data_max_ - scaler.data_min_
model = joblib.load("best_xgboost_model (2).pkl")

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
