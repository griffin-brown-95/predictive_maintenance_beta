import pandas as pd
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the entire pipeline at the start
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Assume the incoming data is a JSON payload

    # Ensure data is in the correct format (e.g., as a DataFrame)
    # Assuming 'data' is a dictionary with feature names as keys
    input_data = pd.DataFrame([data['features']])

    # The model (pipeline) will automatically handle preprocessing and prediction
    prediction = model.predict(input_data)
    
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
