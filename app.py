from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model.pkl')

# Input features (same order as training)
columns = ['age', 'job', 'marital', 'education', 'default', 'balance',
           'housing', 'loan', 'contact', 'day', 'month', 'duration',
           'campaign', 'pdays', 'previous', 'poutcome']

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            input_data = [request.form[col] for col in columns]
            df = pd.DataFrame([input_data], columns=columns)
            prediction = model.predict(df)[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template('index.html', columns=columns, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
