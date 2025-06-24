from flask import Flask, request, jsonify, render_template
import xgboost as xgb
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the saved XGBoost model
model = joblib.load("best_xgboost_model.pkl")

# Manually recreate the MinMaxScaler
scaler = MinMaxScaler()
scaler.data_min_ = np.array([18., 0., 0.])
scaler.data_max_ = np.array([90., 1., 1.])
scaler.data_range_ = scaler.data_max_ - scaler.data_min_
scaler.scale_ = 1 / scaler.data_range_
scaler.min_ = -scaler.data_min_ * scaler.scale_

@app.route('/')
def home():
    return render_template('index.html')  # Ensure this exists in templates/

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form or JSON
        if request.form:
            input_data = [float(request.form.get(feat)) for feat in request.form]
        else:
            input_data = request.get_json(force=True)
            input_data = [input_data[feat] for feat in input_data]

        # Convert to numpy array
        data = np.array(input_data).reshape(1, -1)

        # Normalize the input using manually created scaler
        normalized_data = scaler.transform(data)

        # Predict
        prediction = model.predict(normalized_data)
        result = int(prediction[0])

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
