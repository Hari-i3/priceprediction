from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = load_model('model/rnn_model.h5')
scaler = joblib.load('model/scaler.save')

# Load dataset
data = pd.read_csv('data/sample.csv')
data['date'] = pd.to_datetime(data['date'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    commodity = request.form['commodity']
    date = request.form['date']

    try:
        df = data[data['commodity'] == commodity]
        df.sort_values('date', inplace=True)
        prices = df['price'].values[-30:]
        
        if len(prices) < 30:
            return jsonify({'error': 'Not enough data for prediction.'})

        scaled = scaler.transform(prices.reshape(-1, 1))
        input_seq = scaled.reshape(1, 30, 1)
        pred_scaled = model.predict(input_seq)
        predicted_price = scaler.inverse_transform(pred_scaled)[0][0]

        return jsonify({
            'date': date,
            'commodity': commodity,
           'predicted_price': float(round(predicted_price, 2))

        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
