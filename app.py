from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return 'Adherence Prediction API is running!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array([list(data.values())])
    prediction = model.predict(features)
    return jsonify({'adherence_prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
