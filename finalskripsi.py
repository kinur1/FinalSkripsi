import streamlit as st
import requests
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from datetime import date

# === CONFIG ===
API_KEY = "45G1G2AY7W8KD3S5"
BASE_URL = "https://www.alphavantage.co/query"

# === Helper: ambil data dari Alpha Vantage ===
@st.cache_data(ttl=60*60)
def fetch_alpha_vantage(symbol: str):
    params = {
        "function": "DIGITAL_CURRENCY_DAILY",
        "symbol": symbol,
        "market": "USD",
        "apikey": API_KEY
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    data = r.json()
    if "Time Series (Digital Currency Daily)" not in data:
        return pd.DataFrame()
    ts = data["Time Series (Digital Currency Daily)"]
    records = []
    for dt_str, values in ts.items():
        try:
            records.append({
                "Date": dt_str,
                "Close": float(values.get("4a. close (USD)", 0))
            })
        except Exception:
            continue
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

# === Title ===
st.title("📈 Prediksi Harga Cryptocurrency dengan LSTM (Alpha Vantage)")
st.write("Aplikasi ini memprediksi harga penutupan cryptocurrency menggunakan model LSTM.")

# === Options ===
valid_time_steps = [25, 50, 75, 100]
valid_epochs = [12, 25, 50, 100]

col1, col2 = st.columns(2)
with col1:
    time_step = st.radio("⏳ Time Step", options=valid_time_steps, index=3)
with col2:
    epoch_option = st.radio("🔄 Jumlah Epoch", options=valid_epochs, index=1)

# === Input tanggal ===
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("📅 Tanggal Mulai", date(2020, 1, 1))
with col2:
    end_date = st.date_input("📅 Tanggal Akhir", date(2024, 1, 1))

# === Pilih aset ===
asset_mapping = {"Bitcoin": "BTC", "Ethereum": "ETH"}
asset_name = st.radio("💰 Pilih Aset", list(asset_mapping.keys()))
ticker = asset_mapping[asset_name]

if st.button("🚀 Jalankan Prediksi", disabled=(start_date >= end_date)):

    # === Ambil data ===
    st.write(f"📥 Mengambil data harga {asset_name} ({ticker}) dari Alpha Vantage...")
    df = fetch_alpha_vantage(ticker)

    # filter range tanggal
    mask = (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
    df = df.loc[mask].reset_index(drop=True)

    if df.empty:
        st.error(f"⚠️ Tidak ada data {asset_name} ({ticker}) pada rentang {start_date} s/d {end_date}.")
        st.stop()

    # === Plot harga asli ===
    st.write(f"### 📊 Histori Harga Penutupan {asset_name}")
    fig = px.line(df, x="Date", y="Close", title=f"Histori Harga {asset_name}")
    st.plotly_chart(fig)

    # === Preprocessing ===
    closedf = df[["Close"]]
    scaler = MinMaxScaler(feature_range=(0, 1))
    closedf = scaler.fit_transform(np.array(closedf).reshape(-1, 1))

    training_size = int(len(closedf) * 0.90)
    train_data, test_data = closedf[:training_size], closedf[training_size:]

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # === Build Model ===
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1), activation="relu"),
        LSTM(50, return_sequences=False, activation="relu"),
        Dense(1)
    ])
    model.compile(loss="mean_squared_error", optimizer="adam")

    # === Train ===
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=epoch_option, batch_size=32, verbose=1)

    # === Prediction ===
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
    original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

    # === Evaluation ===
    train_rmse = math.sqrt(mean_squared_error(original_ytrain, train_predict))
    test_rmse = math.sqrt(mean_squared_error(original_ytest, test_predict))
    train_mape = np.mean(np.abs((original_ytrain - train_predict) / original_ytrain)) * 100
    test_mape = np.mean(np.abs((original_ytest - test_predict) / original_ytest)) * 100

    st.write("### 📊 Metrik Evaluasi")
    st.write(f"**✅ RMSE (Training):** {train_rmse}")
    st.write(f"**✅ RMSE (Testing):** {test_rmse}")
    st.write(f"**📉 MAPE (Training):** {train_mape:.2f}%")
    st.write(f"**📉 MAPE (Testing):** {test_mape:.2f}%")

    # === Hasil Prediksi ===
    result_df = pd.DataFrame({
        "Date": df.iloc[time_step+1:time_step+1+len(train_predict)+len(test_predict)]["Date"].values,
        "Original_Close": np.concatenate([original_ytrain.flatten(), original_ytest.flatten()]),
        "Predicted_Close": np.concatenate([train_predict.flatten(), test_predict.flatten()])
    })

    st.write(f"### 🔮 Prediksi Harga {asset_name}")
    fig = px.line(result_df, x="Date", y=["Original_Close", "Predicted_Close"],
                  labels={"value": "Harga", "Date": "Tanggal"})
    st.plotly_chart(fig)

    st.write("### 📊 Hasil Prediksi (DataFrame)")
    st.dataframe(result_df)
