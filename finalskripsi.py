import streamlit as st
import yfinance as yf
import tensorflow as tf
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import plotly.express as px
import datetime

# Streamlit app layout
st.title("📈 Prediksi Harga Cryptocurrency dengan LSTM")
st.write("Aplikasi ini memprediksi harga penutupan cryptocurrency menggunakan model LSTM.")

# Define valid values
valid_time_steps = [25, 50, 75, 100]
valid_epochs = [12, 25, 50, 100]
default_time_step = 100
default_epoch = 25
default_asset = 'BITCOIN'

# Initialize session state to avoid re-running everything
if 'model_ran' not in st.session_state:
    st.session_state.model_ran = False

# Input pengaturan model menggunakan radio buttons dalam kolom
col1, col2 = st.columns(2)
with col1:
    time_step = st.radio("⏳ Time Step", options=valid_time_steps, index=valid_time_steps.index(default_time_step))
with col2:
    epoch_option = st.radio("🔄 Jumlah Epoch", options=valid_epochs, index=valid_epochs.index(default_epoch))

# Input pengaturan data
start_date = st.date_input("📅 Tanggal Mulai", pd.to_datetime("2020-01-01"))
end_date = st.date_input("📅 Tanggal Akhir", pd.to_datetime("2024-01-01"))

# Pilihan aset
asset_name_display = st.radio("💰 Pilih Aset", options=['BITCOIN', 'ETHEREUM'], index=['BITCOIN', 'ETHEREUM'].index(default_asset))

# Check if all inputs are valid
if st.button("🚀 Jalankan Prediksi"):

    # Validation
    if start_date >= end_date:
        st.error("⚠️ Tanggal akhir harus lebih besar dari tanggal mulai.")
        st.stop()

    # Batasi end_date ke hari ini
    today = datetime.date.today()
    if end_date > today:
        st.warning(f"Tanggal akhir ({end_date}) melebihi data yang tersedia. Otomatis diganti ke {today}.")
        end_date = today

    # Mapping aset → ticker Yahoo Finance
    asset_mapping = {
        'BITCOIN': 'BTC-USD',
        'ETHEREUM': 'ETH-USD'
    }
    asset = asset_mapping[asset_name_display]

    # Ambil data dari Yahoo Finance (tambahkan +1 hari ke end_date)
    st.write(f"📥 Mengambil data harga {asset_name_display} ({asset}) dari Yahoo Finance...")
    df = yf.download(asset, start=start_date, end=end_date + datetime.timedelta(days=1))
    df.reset_index(inplace=True)

    # Cek data kosong
    if df.empty or len(df) <= time_step:
        st.error(f"⚠️ Data untuk {asset} tidak ditemukan pada rentang {start_date} s/d {end_date}, "
                 "atau jumlah data tidak cukup untuk membuat dataset.")
        st.stop()

    # Visualisasi data
    st.write(f"### 📊 Histori Harga Penutupan {asset_name_display}")
    fig = px.line(df, x='Date', y='Close', title=f'Histori Harga Penutupan {asset_name_display}')
    st.plotly_chart(fig)

    # Preprocessing
    closedf = df[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    closedf = scaler.fit_transform(np.array(closedf).reshape(-1, 1))

    # Split data
    training_size = int(len(closedf) * 0.90)
    train_data, test_data = closedf[0:training_size, :], closedf[training_size:len(closedf), :1]

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

    # Reshape data
    if len(X_test) == 0 or len(X_train) == 0:
        st.error("⚠️ Data tidak cukup untuk membuat dataset dengan time_step yang dipilih.")
        st.stop()

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(time_step, 1), activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=epoch_option, batch_size=32, verbose=1)

    # Predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transform predictions
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
    original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Evaluation metrics
    train_rmse = math.sqrt(mean_squared_error(original_ytrain, train_predict))
    test_rmse = math.sqrt(mean_squared_error(original_ytest, test_predict))
    train_mape = np.mean(np.abs((original_ytrain - train_predict) / original_ytrain)) * 100
    test_mape = np.mean(np.abs((original_ytest - test_predict) / original_ytest)) * 100

    # Save model state
    st.session_state.model_ran = True
    st.session_state.df = df
    st.session_state.train_predict = train_predict
    st.session_state.test_predict = test_predict
    st.session_state.original_ytrain = original_ytrain
    st.session_state.original_ytest = original_ytest
    st.session_state.time_step = time_step

    # Display metrics
    st.write("### 📊 Metrik Evaluasi")
    st.write(f"**RMSE (Training):** {train_rmse:.2f}")
    st.write(f"**RMSE (Testing):** {test_rmse:.2f}")
    st.write(f"**MAPE (Training):** {train_mape:.2f}%")
    st.write(f"**MAPE (Testing):** {test_mape:.2f}%")
