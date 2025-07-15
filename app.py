from flask import Flask, request, render_template
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from lime.lime_tabular import LimeTabularExplainer
import shap
import pandas as pd

app = Flask(__name__)

# Load your trained XGBClassifier (not raw Booster)
model = xgb.XGBClassifier()
model.load_model("xgb_booster.json")  # This works if you saved your classifier with `.save_model`
print("✅ XGBClassifier loaded.")

# Recreate scaler
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

        # Wrap in DataFrame so feature names are always correct
        input_df = pd.DataFrame([user_input])[feature_order]

        num_idx = [f for f in numerical_features]
        input_df[num_idx] = scaler.transform(input_df[num_idx])

        prob = model.predict_proba(input_df)[0][1]
        pred = int(prob >= 0.5)

        # === LIME ===
        explainer = LimeTabularExplainer(
            training_data=np.random.rand(100, len(feature_order)),  # fake, replace with your X_train
            feature_names=feature_order,
            class_names=['No', 'Yes'],
            mode='classification'
        )
        lime_exp = explainer.explain_instance(
            input_df.iloc[0].values,
            model.predict_proba,
            num_features=5
        )
        lime_list = lime_exp.as_list()

        # === SHAP ===
        explainer_shap = shap.TreeExplainer(model)
        shap_values = explainer_shap.shap_values(input_df)
        shap_vals = dict(zip(feature_order, shap_values[0].tolist()))

        return render_template(
            'result.html',
            prediction=pred,
            probability=round(prob * 100, 2),
            lime_list=lime_list,
            shap_vals=shap_vals
        )

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
