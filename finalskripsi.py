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

    # Reshape data
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build LSTM Model (Enhanced)
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1), activation="relu"),
        LSTM(50, return_sequences=False, activation="relu"),
        Dense(1)
    ])
    model.compile(loss="mean_squared_error", optimizer="adam")

    # Train Model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch_option, batch_size=32, verbose=1)

    # Predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transform
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
    original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Evaluation Metrics
    train_rmse = math.sqrt(mean_squared_error(original_ytrain, train_predict))
    test_rmse = math.sqrt(mean_squared_error(original_ytest, test_predict))
    train_mape = np.mean(np.abs((original_ytrain - train_predict) / original_ytrain)) * 100
    test_mape = np.mean(np.abs((original_ytest - test_predict) / original_ytest)) * 100

    # Save Model State
    st.session_state.update({
        'model_ran': True, 'df': df,
        'train_predict': train_predict, 'test_predict': test_predict,
        'original_ytrain': original_ytrain, 'original_ytest': original_ytest,
        'time_step': time_step, 'num_test_days': len(test_predict)
    })

    # Display metrics
    st.write("### 📊 Metrik Evaluasi")
    st.write(f"**✅ RMSE (Training):** {train_rmse}")
    st.write(f"**✅ RMSE (Testing):** {test_rmse}")
    st.write(f"**📉 MAPE (Training):** {train_mape:.2f}%")
    st.write(f"**📉 MAPE (Testing):** {test_mape:.2f}%")

# Menampilkan hasil prediksi setelah model dijalankan
if st.session_state.model_ran:
    df = st.session_state.df
    train_predict = st.session_state.train_predict
    test_predict = st.session_state.test_predict
    original_ytrain = st.session_state.original_ytrain
    original_ytest = st.session_state.original_ytest

    # DataFrame Prediksi
    predict_dates = df['Date'][st.session_state.time_step+1:st.session_state.time_step+1+len(train_predict)+len(test_predict)]
    result_df = pd.DataFrame({
        'Date': df.iloc[time_step+1:len(train_predict)+len(test_predict)+time_step+1]['Date'].values,
    'Original_Close': np.concatenate([original_ytrain.flatten(), original_ytest.flatten()]),
    'Predicted_Close': np.concatenate([train_predict.flatten(), test_predict.flatten()])
    })

    # Plot hasil prediksi
    st.write(f"### 🔮 Prediksi Harga {asset_name_display}")
    fig = px.line(result_df, x='Date', y=['Original_Close', 'Predicted_Close'], labels={'value': 'Harga', 'Date': 'Tanggal'})
    st.plotly_chart(fig)

    # Tampilkan DataFrame
    st.write("### 📊 Hasil Prediksi")
    st.write(result_df)
