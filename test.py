import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, flash
import os

# Load dataset and preprocess
try:
    dataset = pd.read_csv('dataset/upi_fraud_dataset.csv', index_col=0)
    x = dataset.iloc[:, :10].values
    y = dataset.iloc[:, 10].values
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
except Exception as e:
    print(f"Error loading or processing dataset: {e}")

# Load the trained model
try:
    model = tf.keras.models.load_model('model/project_model1.h5')
except Exception as e:
    print(f"Error loading model: {e}")

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for flashing messages

@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/preview', methods=["POST"])
def preview():
    try:
        if request.method == 'POST':
            dataset = request.files['datasetfile']
            df = pd.read_csv(dataset, encoding='unicode_escape')
            df.set_index('Id', inplace=True)
            return render_template("preview.html", df_view=df)
    except Exception as e:
        flash(f"Error processing the file: {e}")
        return redirect(url_for('upload'))

@app.route('/prediction1', methods=['GET'])
def prediction1():
    return render_template('index.html')

@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Extracting data from form
        trans_datetime = pd.to_datetime(request.form.get("trans_datetime"))
        dob = pd.to_datetime(request.form.get("dob"))
        
        # Feature extraction for the model
        v1 = trans_datetime.hour
        v2 = trans_datetime.day
        v3 = trans_datetime.month
        v4 = trans_datetime.year
        v5 = int(request.form.get("category"))
        v6 = float(request.form.get("card_number"))
        v7 = np.round((trans_datetime - dob) // np.timedelta64(1, 'Y'))  # Age
        v8 = float(request.form.get("trans_amount"))
        v9 = int(request.form.get("state"))
        v10 = int(request.form.get("zip"))
        
        # Prepare input for the model
        x_test = np.array([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10])
        x_test_scaled = scaler.transform([x_test])
        
        # Prediction
        y_pred = model.predict(x_test_scaled)
        result = "VALID TRANSACTION" if y_pred[0][0] <= 0.5 else "FRAUD TRANSACTION"
        
        return render_template('result.html', OUTPUT=result)
    
    except Exception as e:
        flash(f"Error during prediction: {e}")
        return redirect(url_for('prediction1'))

if __name__ == "__main__":
    # Ensure the application runs on a desired host and port
    app.run(debug=True, host='0.0.0.0', port=5000)
