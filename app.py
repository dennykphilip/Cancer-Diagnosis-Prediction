from flask import Flask, render_template, request
import numpy as np
import joblib

# Initialize Flask application
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('cancer_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')


# Home route
@app.route('/')
def home():
    return render_template('index.html')


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input data from form
    input_features = [float(request.form[f'feature{i}']) for i in range(1, 31)]  # Collect 30 features
    input_data = np.array(input_features).reshape(1, -1)

    # Scale input data
    input_data_scaled = scaler.transform(input_data)

    # Predict using the loaded model
    prediction = model.predict(input_data_scaled)

    # Convert numerical prediction to categorical diagnosis
    output = 'Malignant (M)' if prediction[0] == 1 else 'Benign (B)'

    return render_template('index.html', prediction_text=f'The predicted diagnosis is: {output}')


# Run the application
if __name__ == "__main__":
    app.run(debug=True)
