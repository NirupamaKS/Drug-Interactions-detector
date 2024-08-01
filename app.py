from flask import Flask, render_template, request
import pickle
from utils import custom_tokenizer

app = Flask(__name__)

# Load the trained model
with open('drug_interaction_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    drug1 = request.form['drug1']
    drug2 = request.form['drug2']
    
    input_data = f"{drug1}{drug2}"
    prediction = model.predict([input_data])[0]
    
    return render_template('result.html', drug1=drug1, drug2=drug2, interaction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
