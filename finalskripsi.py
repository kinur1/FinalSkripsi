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

# Streamlit app layout
st.title("Prediksi Harga Cryptocurrency dengan LSTM")
st.write("Aplikasi ini memprediksi harga penutupan cryptocurrency menggunakan model LSTM.")

# Define valid values
valid_time_steps = [25, 50, 75, 100]
valid_epochs = [12, 25, 50, 100]
default_time_step = 100
default_epoch = 25
default_asset = 'BITCOIN'

# Initialize session state
if 'model_ran' not in st.session_state:
    st.session_state.model_ran = False

# Input model params
col1, col2 = st.columns(2)
with col1:
    time_step = st.radio("Time Step", options=valid_time_steps, index=valid_time_steps.index(default_time_step))
with col2:
    epoch_option = st.radio("Jumlah Epoch", options=valid_epochs, index=valid_epochs.index(default_epoch))

# Input data range
start_date = st.date_input("Tanggal Mulai", pd.to_datetime("2020-01-01"))
end_date = st.date_input("Tanggal Akhir", pd.to_datetime("2024-01-01"))

# Pilihan aset
asset_name_display = st.radio("Pilih Aset", options=['BITCOIN', 'ETHEREUM'], index=['BITCOIN', 'ETHEREUM'].index(default_asset))

if st.button("Jalankan Prediksi"):

    # Mapping aset ke ticker
    asset_mapping = {
        'BITCOIN': 'BTC-USD',
        'ETHEREUM': 'ETH-USD'
    }
    asset = asset_mapping[asset_name_display]

    # Validasi ticker terlebih dahulu
    try:
        df_check = yf.download(asset, period="5d")
        if df_check.empty:
            st.error(f"❌ Ticker {asset} tidak valid atau tidak tersedia di Yahoo Finance.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Gagal memeriksa ticker {asset}: {e}")
        st.stop()

    # Ambil data sesuai rentang
    st.write(f"Mengambil data harga {asset_name_display} ({asset}) dari Yahoo Finance...")
    df = yf.download(asset, start=start_date, end=end_date)
    df.reset_index(inplace=True)

    # Validasi data kosong
    if df.empty:
        st.error(f"⚠️ Tidak ada data untuk {asset} pada rentang {start_date} s/d {end_date}. "
                 f"Silakan pilih tanggal lain.")
        st.stop()

    # Visualisasi data
    st.write(f"### Histori Harga Penutupan {asset_name_display}")
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

    # Reshape
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(time_step, 1), activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")

    # Train
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=epoch_option, batch_size=32, verbose=1)

    # Predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transform
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
    original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Metrics
    train_rmse = math.sqrt(mean_squared_error(original_ytrain, train_predict))
    test_rmse = math.sqrt(mean_squared_error(original_ytest, test_predict))
    train_mape = np.mean(np.abs((original_ytrain - train_predict) / original_ytrain)) * 100
    test_mape = np.mean(np.abs((original_ytest - test_predict) / original_ytest)) * 100

    st.session_state.model_ran = True
    st.session_state.df = df
    st.session_state.train_predict = train_predict
    st.session_state.test_predict = test_predict
    st.session_state.original_ytrain = original_ytrain
    st.session_state.original_ytest = original_ytest
    st.session_state.time_step = time_step
    st.session_state.num_test_days = len(test_predict)

    # Display metrics
    st.write("### Metrik Evaluasi")
    st.write(f"**RMSE (Training Data):** {train_rmse}")
    st.write(f"**RMSE (Testing Data):** {test_rmse}")
    st.write(f"**MAPE (Training Data):** {train_mape}%")
    st.write(f"**MAPE (Testing Data):** {test_mape}%")

    st.write("### Detail Lain")
    st.write(f"**Jumlah Data Training:** {len(X_train)}")
    st.write(f"**Jumlah Data Testing:** {len(X_test)}")
    st.write(f"**Jumlah Epoch yang Dipilih:** {epoch_option}")
    st.write(f"**Time Step yang Dipilih:** {time_step}")
    st.write(f"**Jumlah Hari yang Diprediksi:** {st.session_state.num_test_days}")

# Jika model sudah dijalankan, tampilkan hasil prediksi
if st.session_state.model_ran:
    look_back = st.session_state.time_step
    df = st.session_state.df
    train_predict = st.session_state.train_predict
    test_predict = st.session_state.test_predict
    original_ytrain = st.session_state.original_ytrain
    original_ytest = st.session_state.original_ytest

    trainPredictPlot = np.empty_like(df[['Close']])
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

    testPredictPlot = np.empty_like(df[['Close']])
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df) - 1, :] = test_predict

    plotdf = pd.DataFrame({
        'Date': df['Date'].values,
        'Original_Close': df['Close'].values,
        'Train_Predicted_Close': trainPredictPlot.reshape(1, -1)[0][:len(df)],
        'Test_Predicted_Close': testPredictPlot.reshape(1, -1)[0][:len(df)]
    })

    st.write(f"### Perbandingan Harga Penutupan Asli vs Prediksi untuk {asset_name_display}")
    fig = px.line(plotdf, x='Date', y=['Original_Close', 'Train_Predicted_Close', 'Test_Predicted_Close'],
                  labels={'value': 'Harga', 'Date': 'Tanggal'},
                  title=f'Harga Penutupan Asli vs Prediksi untuk {asset_name_display}')
    st.plotly_chart(fig)

    # Tampilkan DataFrame hasil prediksi
    result_df = pd.DataFrame({
        'Date': df['Date'][st.session_state.time_step + 1:st.session_state.time_step + 1 + len(train_predict) + len(test_predict)],
        'Original_Close': np.concatenate([st.session_state.original_ytrain.flatten(), st.session_state.original_ytest.flatten()]),
        'Predicted_Close': np.concatenate([train_predict.flatten(), test_predict.flatten()])
    })
    result_df.reset_index(drop=True, inplace=True)

    st.write("### Hasil Prediksi dalam DataFrame")
    st.write(result_df)
