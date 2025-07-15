from flask import Flask, request, render_template
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from lime.lime_tabular import LimeTabularExplainer
import shap
from sklearn.inspection import permutation_importance
import pandas as pd

app = Flask(__name__)

# Booster
booster = xgb.Booster()
booster.load_model("xgb_booster.json")
print("✅ Booster loaded.")

# Scaler
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

        dmatrix = xgb.DMatrix(input_vector, feature_names=feature_order)
        prob = booster.predict(dmatrix)
        pred = int(prob[0] >= 0.5)

        # === 1️⃣ LIME ===
        explainer = LimeTabularExplainer(
            training_data=np.random.rand(100, len(feature_order)),  # dummy, replace with your X_train
            feature_names=feature_order,
            class_names=['No', 'Yes'],
            mode='classification'
        )
        # You’d normally pass the model.predict_proba, so here we wrap Booster:
        lime_exp = explainer.explain_instance(
            input_vector[0],
            lambda x: booster.predict(xgb.DMatrix(x, feature_names=feature_order)).reshape(-1, 1),
            num_features=5
        )
        lime_list = lime_exp.as_list()

        # === 2️⃣ SHAP ===
        shap.initjs()
        booster_model = xgb.XGBClassifier()
        booster_model._Booster = booster
        explainer_shap = shap.TreeExplainer(booster)
        shap_values = explainer_shap.shap_values(input_vector)
        shap_vals = dict(zip(feature_order, shap_values[0].tolist()))

        # === 3️⃣ Permutation ===
        # For this example you’d use your training set — here we fake with random for demo
        X_dummy = np.random.rand(50, len(feature_order))
        y_dummy = np.random.randint(0, 2, 50)
        perm_importance = permutation_importance(
            booster_model, X_dummy, y_dummy, n_repeats=5, random_state=0
        )
        perm_result = dict(zip(feature_order, perm_importance.importances_mean.tolist()))

        return render_template(
            'result.html',
            prediction=pred,
            probability=round(prob[0] * 100, 2),
            lime_list=lime_list,
            shap_vals=shap_vals,
            perm_result=perm_result
        )

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
