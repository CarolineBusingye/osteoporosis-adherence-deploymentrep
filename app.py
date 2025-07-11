import xgboost as xgb
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load only the booster
booster = xgb.Booster()
booster.load_model("xgb_booster.json")

scaler = MinMaxScaler()
scaler.data_min_ = np.array([18., 0.])
scaler.data_max_ = np.array([90., 1.])
scaler.data_range_ = scaler.data_max_ - scaler.data_min_
scaler.scale_ = 1 / scaler.data_range_
scaler.min_ = -scaler.data_min_ * scaler.scale_

FEATURE_NAMES = [
    'Age', 'Race/Ethnicity', 'Body Weight', 'Osteoporosis',
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
        if request.form:
            user_input = {}
            for feat in feature_order:
                val = request.form.get(feat)
                if feat in numerical_features:
                    user_input[feat] = float(val) if val else 0.0
                else:
                    user_input[feat] = int(val) if val else 0
        else:
            data = request.get_json(force=True)
            user_input = {feat: data.get(feat, 0) for feat in feature_order}

        input_vector = np.array([user_input[feat] for feat in feature_order]).reshape(1, -1)

        num_idx = [feature_order.index(f) for f in numerical_features]
        input_vector[:, num_idx] = scaler.transform(input_vector[:, num_idx])

        dmatrix = xgb.DMatrix(input_vector, feature_names=FEATURE_NAMES)

        proba = booster.predict(dmatrix)
        prediction = int(proba[0] >= 0.5)

        return jsonify({
            'mode': 'Booster',
            'prediction': prediction,
            'probability': float(proba[0])
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
