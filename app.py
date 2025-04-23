from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate and get the form data
        age = request.form['age']
        if not age or not age.isdigit():
            raise ValueError("Invalid age. Please enter a valid number.")

        age = float(age)

        # Validate and extract systolic and diastolic BP
        bp = request.form['bp']
        if not bp or '/' not in bp:
            raise ValueError("Invalid blood pressure. Please enter in the format 120/80.")
        
        bp_values = bp.split('/')
        if len(bp_values) != 2 or not all(val.isdigit() for val in bp_values):
            raise ValueError("Invalid blood pressure. Please enter valid systolic and diastolic values.")

        sys_bp = float(bp_values[0])
        dia_bp = float(bp_values[1])

        # Validate creatinine
        creatinine = request.form['creatinine']
        if not creatinine or not creatinine.replace('.', '', 1).isdigit():
            raise ValueError("Invalid creatinine level. Please enter a valid number.")

        creatinine = float(creatinine)

        # Validate family history and diabetes
        family_history = request.form['family_history']
        diabetes = request.form['diabetes']

        if family_history not in ['Yes', 'No']:
            raise ValueError("Invalid value for family history.")
        
        if diabetes not in ['Yes', 'No']:
            raise ValueError("Invalid value for diabetes.")

        family_history = 1 if family_history == 'Yes' else 0
        diabetes = 1 if diabetes == 'Yes' else 0

        # Prepare input features
        input_features = np.array([[age, sys_bp, dia_bp, creatinine, family_history, diabetes]])

        # Make prediction
        prediction = model.predict(input_features)[0]
        result = "Dialysis Needed" if prediction == 1 else "Dialysis Not Needed"
        
        return render_template('index.html', result=result)

    except Exception as e:
        return render_template('index.html', result=f"Error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
