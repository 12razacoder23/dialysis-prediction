from flask import Flask, render_template, request
import pandas as pd
import joblib  # or use pickle

# Initialize Flask app
app = Flask(__name__)

# Load your model ONCE
model = joblib.load('model.pkl')  # Make sure model.pkl exists in your project folder

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        bp = float(request.form['bp'])
        creatinine = float(request.form['creatinine'])

        input_data = pd.DataFrame([[age, bp, creatinine]], columns=['age', 'bp', 'creatinine'])
        prediction = model.predict(input_data)[0]
        result = "Yes dialysis needed" if prediction == 'Yes' or prediction == 1 else "No dialysis not needed"

        return render_template('index.html', result=result)

    except Exception as e:
        print("Error:", e)
        return render_template('index.html', result="Error in prediction")

if __name__ == '__main__':
    app.run(debug=True)
