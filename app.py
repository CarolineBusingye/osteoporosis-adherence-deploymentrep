from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load model
model = joblib.load("best_xgboost_model (2).pkl")

# Recreate MinMaxScaler for ['Age', 'Osteoporosis']
scaler = MinMaxScaler()
scaler.data_min_ = np.array([18., 0.])
scaler.data_max_ = np.array([90., 1.])
scaler.data_range_ = scaler.data_max_ - scaler.data_min_
scaler.scale_ = 1 / scaler.data_range_
scaler.min_ = -scaler.data_min_ * scaler.scale_

# Updated feature list (excluding 'Adherence')
feature_order = [
    'Age', 'Osteoporosis',
    'Gender_Female', 'Gender_Male',
    'Hormonal Changes_Normal', 'Hormonal Changes_Postmenopausal',
    'Family History_No', 'Family History_Yes',
    'Smoking_No', 'Smoking_Yes',
    'Alcohol Consumption_Moderate',
    'Medical Conditions_Hyperthyroidism', 'Medical Conditions_Rheumatoid Arthritis',
    'Medications_Corticosteroids',
    'Prior Fractures_No', 'Prior Fractures_Yes'
]

numerical_features = ['Age', 'Osteoporosis']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        if request.form:
            user_input = {k: float(request.form.get(k)) for k in feature_order}
        else:
            user_input = request.get_json(force=True)

        # Order inputs correctly
        input_vector = np.array([user_input[feat] for feat in feature_order]).reshape(1, -1)

        # Scale numerical values
        input_vector_scaled = input_vector.copy()
        indices = [feature_order.index(f) for f in numerical_features]
        input_vector_scaled[:, indices] = scaler.transform(input_vector[:, indices])

        # Predict
        prediction = model.predict(input_vector_scaled)
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
