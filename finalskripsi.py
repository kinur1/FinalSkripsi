import streamlit as st
import yfinance as yf
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Title
st.title("📈 Prediksi Harga Cryptocurrency dengan LSTM")
st.write("Aplikasi ini memprediksi harga penutupan cryptocurrency menggunakan model LSTM.")

# Valid options
valid_time_steps = [25, 50, 75, 100]
valid_epochs = [12, 25, 50, 100]
default_time_step = 100
default_epoch = 25
default_asset = 'BITCOIN'

# Session state
if 'model_ran' not in st.session_state:
    st.session_state.model_ran = False

# Input settings
col1, col2 = st.columns(2)
with col1:
    time_step = st.radio("⏳ Time Step", options=valid_time_steps, index=valid_time_steps.index(default_time_step))
with col2:
    epoch_option = st.radio("🔄 Jumlah Epoch", options=valid_epochs, index=valid_epochs.index(default_epoch))

# Date selection
start_date = st.date_input("📅 Tanggal Mulai", pd.to_datetime("2020-01-01"))
end_date = st.date_input("📅 Tanggal Akhir", pd.to_datetime("2024-01-01"))

# Asset selection
asset_name_display = st.radio("💰 Pilih Aset", options=['BITCOIN', 'ETHEREUM'], index=0)

# Validasi Input
is_valid = (start_date < end_date)

# Run Prediction Button
if st.button("🚀 Jalankan Prediksi", disabled=not is_valid):
    
    # Mapping assets
    asset_mapping = {'BITCOIN': 'BTC-USD', 'ETHEREUM': 'ETH-USD'}
    asset = asset_mapping[asset_name_display]

    # Fetch data
    st.write(f"📥 Mengambil data harga {asset_name_display} ({asset}) dari Yahoo Finance...")
    df = yf.download(asset, start=start_date, end=end_date)
    df = df.reset_index()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Validasi jumlah data
    if len(df) <= time_step + 1:
        st.error(f"⚠️ Data terlalu sedikit ({len(df)} baris) untuk time_step={time_step}. "
                 f"Coba perpanjang rentang tanggal atau kurangi time_step.")
    else:
        # Plot harga asli
        st.write(f"### 📊 Histori Harga Penutupan {asset_name_display}")
        fig = px.line(df, x='Date', y='Close', title=f'Histori Harga {asset_name_display}')
        st.plotly_chart(fig)

        # Preprocessing
        closedf = df[['Close']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        closedf = scaler.fit_transform(np.array(closedf).reshape(-1, 1))

        # Split data
        training_size = int(len(closedf) * 0.90)
        train_data, test_data = closedf[:training_size], closedf[training_size:]

        # Function to create dataset
        def create_dataset(dataset, time_step=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - time_step - 1):
                a = dataset[i:(i + time_step), 0]
                dataX.append(a)
                dataY.append(dataset[i + time_step, 0])
            return np.array(dataX), np.array(dataY)

        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)

        # Cegah error reshape jika data kosong
        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            st.error("⚠️ Dataset training atau testing kosong. "
                     "Perbesar rentang tanggal atau kurangi time_step.")
        else:
            # Reshape data
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            # Build LSTM Model (Enhanced)
            model = Sequential([
