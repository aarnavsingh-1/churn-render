from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form
    try:
        # Retrieve values in the same order as the model expects
        features = [
            int(request.form['CreditScore']),
            int(request.form['Age']),
            int(request.form['Tenure']),
            float(request.form['Balance']),
            int(request.form['NumOfProducts']),
            int(request.form['HasCrCard']),
            int(request.form['IsActiveMember']),
            float(request.form['EstimatedSalary']),
            int(request.form['Geography_Germany']),
            int(request.form['Geography_Spain']),
            int(request.form['Gender_Male'])
        ]

        # Convert to the appropriate format for the model
        final_features = [np.array(features)]
        
        # Make prediction
        prediction = model.predict(final_features)
        output = 'Churned' if prediction[0] == 1 else 'Not Churned'

        return render_template('index.html', prediction_text=f'Prediction: {output}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)



