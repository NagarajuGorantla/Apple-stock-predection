# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

# ✅ Load historical data safely
try:
    data = pd.read_csv("AAPL.csv", index_col="Date", parse_dates=True)
    data.index = pd.to_datetime(data.index, errors="coerce")  # Ensure datetime format
    data.dropna(inplace=True)  # Drop rows with invalid dates
except FileNotFoundError:
    st.error("❌ CSV file 'AAPL.csv' not found! Please check the file path.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading CSV file: {e}")
    st.stop()

# ✅ Load trained model safely
try:
    with open("rf_model.pkl", "rb") as file:
        model = pickle.load(file)
    
    if not hasattr(model, "predict"):  # Ensure it's a valid model
        raise ValueError("Loaded file is not a trained ML model!")
except FileNotFoundError:
    st.error("❌ Model file 'rf_model.pkl' not found! Please train and save the model.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ✅ Streamlit UI Configuration
st.set_page_config(page_title="Apple Stock Prediction", layout="wide")
st.title("📈 Apple Stock Price Prediction")
st.sidebar.header("User Input")

# ✅ Date range selection
start_date = st.sidebar.date_input("Select Start Date", data.index.min().date())
end_date = st.sidebar.date_input("Select End Date", data.index.max().date())

# ✅ Ensure valid date range
if start_date >= end_date:
    st.sidebar.error("❌ End date must be after start date.")
    st.stop()

# ✅ Filter data based on user selection
filtered_data = data.loc[start_date:end_date]

# ✅ Display historical data
st.subheader("📊 Historical Stock Prices")
st.line_chart(filtered_data["Close"], use_container_width=True)

# ✅ Predict next 30 days
st.subheader("🔮 Predicted Stock Prices for Next 30 Days")

# Ensure we have enough data for predictions
if len(data) == 0:
    st.error("❌ No valid historical data available for predictions.")
    st.stop()

pred_dates = [data.index[-1] + timedelta(days=i) for i in range(1, 31)]
# Extract last known feature values
last_known_features = data.iloc[-1][["Open", "High", "Low", "Volume"]].values

# Repeat these values for the next 30 days
pred_features = np.tile(last_known_features, (30, 1))  # Shape (30, 4)

try:
    pred_prices = model.predict(pred_features)  # Ensure valid model prediction
except Exception as e:
    st.error(f"❌ Prediction Error: {e}")
    st.stop()

# ✅ Display predictions
df_pred = pd.DataFrame({"Date": pred_dates, "Predicted Price": pred_prices})
st.dataframe(df_pred)

# ✅ Visualization
st.subheader("📉 Prediction Trend")
fig, ax = plt.subplots(figsize=(12, 6))
sns.set_style("darkgrid")
ax.plot(data.index, data["Close"], label="Historical Prices", color="#00b4d8", linewidth=2)
ax.plot(df_pred["Date"], df_pred["Predicted Price"], label="Predicted Prices", color="#ff4c4c", linestyle="dashed", linewidth=2)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Stock Price ($)", fontsize=12)
ax.set_title("Apple Stock Price Forecast", fontsize=16, fontweight="bold")
ax.legend(fontsize=12)
st.pyplot(fig)

st.success(" Predictions generated successfully!")
