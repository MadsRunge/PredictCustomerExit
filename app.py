from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os


current_dir = os.path.dirname(os.path.abspath(__file__))


app = Flask(__name__, template_folder=os.path.join(current_dir, 'templates'))


model = load_model(os.path.join(current_dir, 'customer_churn_model.keras'))

with open(os.path.join(current_dir, 'scaler.pkl'), 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
          
            input_data = [
                float(request.form['CreditScore']),
                float(request.form['Age']),
                float(request.form['Tenure']),
                float(request.form['Balance']),
                float(request.form['NumOfProducts']),
                float(request.form['HasCrCard']),
                float(request.form['IsActiveMember']),
                float(request.form['EstimatedSalary']),
                1 if request.form['Geography'] == 'France' else 0,
                1 if request.form['Geography'] == 'Germany' else 0,
                1 if request.form['Geography'] == 'Spain' else 0,
                1 if request.form['Gender'] == 'Female' else 0,
                1 if request.form['Gender'] == 'Male' else 0
            ]

           
            scaled_input = scaler.transform([input_data])

           
            prediction = model.predict(scaled_input)
            churn_probability = prediction[0][0] * 100

            return render_template('index.html', result=f"Probability of customer churn: {churn_probability:.2f}%")
        
        return render_template('index.html', result=None)
    except Exception as e:
        return f"An error occurred: {str(e)}"

@app.route('/test')
def test():
    return "Flask is working!"

if __name__ == '__main__':
    print(f"Current directory: {current_dir}")
    print(f"Templates folder: {app.template_folder}")
    print(f"Index.html exists: {os.path.exists(os.path.join(app.template_folder, 'index.html'))}")
    app.run(debug=True, host='0.0.0.0', port=2000)
