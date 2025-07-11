from flask import Flask, request, jsonify, render_template
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# ✅ Load only the Booster (version safe)
booster = xgb.Booster()
booster.load_model("xgb_booster.json")
print("✅ Booster loaded.")

# ✅ Recreate your scaler
scaler = MinMaxScaler()
scaler.data_min_ = np.array([18., 0., 1., 40.])
scaler.data_max_ = np.array([90., 1., 5., 120.])
scaler.data_range_ = scaler.data_max_ - scaler.data_min_
scaler.scale_ = 1 / scaler.data_range_
scaler.min_ = -scaler.data_min_ * scaler.scale_

feature_order = [
    'Age', 'Osteoporosis', 'Race/Ethnicity', 'Body Weight',
    'Gender_Female', 'Gender_Male',
    'Hormonal Changes_Normal', 'Hormonal Changes_Postmenopausal',
    'Family History_No', 'Family History_Yes',
    'Smoking_No', 'Smoking_Yes',
    'Alcohol Consumption_Moderate',
    'Medical Conditions_Hyperthyroidism', 'Medical Conditions_Rheumatoid Arthritis',
    'Medications_Corticosteroids',
    'Prior Fractures_No', 'Prior Fractures_Yes'
]

numerical_features = ['Age', 'Osteoporosis', 'Race/Ethnicity', 'Body Weight']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = {}
        for feat in feature_order:
            val = request.form.get(feat)
            if feat in numerical_features:
                user_input[feat] = float(val) if val else 0.0
            else:
                user_input[feat] = int(val) if val else 0

        input_vector = np.array([user_input[f] for f in feature_order]).reshape(1, -1)

        num_idx = [feature_order.index(f) for f in numerical_features]
        input_vector[:, num_idx] = scaler.transform(input_vector[:, num_idx])

        dmatrix = xgb.DMatrix(input_vector)
        prob = booster.predict(dmatrix)
        pred = int(prob[0] >= 0.5)

        return jsonify({
            'prediction': pred,
            'probability': float(prob[0])
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
