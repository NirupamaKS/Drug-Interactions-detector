# app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('drug_interactions.csv')

# Load the model (if you have pre-trained a machine learning model)
model = joblib.load('interaction_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    drug1 = request.form['drug1']
    drug2 = request.form['drug2']
    
    # Add your interaction prediction logic here
    # For example, using a simple lookup in the dataframe:
    interaction = df[(df['drug_1'] == drug1) & (df['drug_2'] == drug2)]
    if not interaction.empty:
        result = interaction['interaction'].values[0]
    else:
        result = "Unknown"

    return jsonify({'interaction': result})

if __name__ == '__main__':
    app.run(debug=True)
