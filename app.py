# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import yfinance as yf
import datetime
from sklearn.preprocessing import MinMaxScaler

# Load models
lr_model = joblib.load('linear_regression_model_1.pkl')
lr_model2 = joblib.load('linear_regression_model.pkl')
lstm_model = load_model('Stock Predictions Model.keras')
scaler = MinMaxScaler(feature_range=(0, 1))

# App UI
st.title("📈 Stock Price Predictor")
stock_symbol = st.text_input("Enter Stock Symbol (e.g., TSLA)", value="TSLA")
model_choice = st.radio("Select Model", ("Linear Regression Model 1", "Linear Regression Model 2", "LSTM Model"))
predict_days = st.slider("Days to Forecast", min_value=1, max_value=100, value=5)

# Fetch stock data
start_date = "2015-01-01"
end_date = datetime.datetime.today().strftime('%Y-%m-%d')
df = yf.download(stock_symbol, start=start_date, end=end_date)
if df.empty or 'Close' not in df.columns:
    st.error("No data found for the given stock symbol and date range. Please try another symbol.")
    st.stop()

print("Head of df:")
print(df.head())

print("Columns:")
print(df.columns)

print("Close column:")
print(df['Close'].head())

print("df[['Close']].values:")
print(df[['Close']].values)

# Fit the scaler on historical closing prices
scaler.fit(df[['Close']].values)


if df.empty:
    st.warning("⚠️ No data found for this symbol.")
else:
    st.subheader(f"📊 Historical Data for {stock_symbol}")
    st.line_chart(df['Close'])

    if st.button("🔮 Predict"):
        if model_choice == "Linear Regression Model 1":
            # Use last row features (e.g., Close, MA10, MA50)
            df['MA10'] = df['Close'].rolling(10).mean()
            df['MA50'] = df['Close'].rolling(50).mean()
            last_row = df[['Close', 'MA10', 'MA50']].dropna().iloc[-1]
            X_input = np.array(last_row).reshape(1, -1)
            future_preds = []
            for _ in range(predict_days):
                pred = lr_model.predict(X_input)[0]
                future_preds.append(pred)
                # Update inputs for next prediction
                X_input = np.array([pred, X_input[0][1], X_input[0][2]]).reshape(1, -1)
            st.success("✅ Predictions (Linear Regression):")
            st.write(future_preds)
        elif model_choice == "Linear Regression Model 2":
            df['MA10'] = df['Close'].rolling(10).mean()
            df['MA50'] = df['Close'].rolling(50).mean()
            last_row = df[['Close', 'MA10', 'MA50']].dropna().iloc[-1]
            X_input = np.array(last_row).reshape(1, -1)
            future_preds = []
            for _ in range(predict_days):
                pred = lr_model2.predict(X_input)[0]
                future_preds.append(pred)
                # Update inputs for next prediction
                X_input = np.array([pred, X_input[0][1], X_input[0][2]]).reshape(1, -1)
            st.success("✅ Predictions (Linear Regression):")
            st.write(future_preds)
        else:  # LSTM
            closing_prices = df[['Close']].values
            scaled_data = scaler.transform(closing_prices)

            past_days = 100
            input_sequence = scaled_data[-past_days:]
            input_sequence = input_sequence.reshape(1, past_days, 1)

            future_preds = []
            for _ in range(predict_days):
                pred_scaled = lstm_model.predict(input_sequence)[0][0]
                future_preds.append(scaler.inverse_transform([[pred_scaled]])[0][0])
                new_input = np.append(input_sequence[0][1:], [[pred_scaled]], axis=0)
                input_sequence = new_input.reshape(1, past_days, 1)

            st.success("✅ Predictions (LSTM):")
            st.write(future_preds)
