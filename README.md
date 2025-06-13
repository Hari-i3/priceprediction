# Price Prediction using RNN (Flask + TensorFlow)

This project predicts future prices of essential food commodities using a Recurrent Neural Network (RNN) model trained on historical price data. The application is built with Flask for serving predictions.

## Features

- Trained RNN model with `SimpleRNN`
- Predicts price for selected commodity (e.g., onion)
- Web interface for user input
- JSON API response

## Run Instructions

```bash
python train_model.py   # Train the model
python app.py           # Run the Flask app
