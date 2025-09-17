import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import math

st.title("🔮 Prediksi Harga Cryptocurrency dengan LSTM")

# Input
asset = st.text_input("Masukkan Ticker (contoh: BTC-USD):", "BTC-USD")
start_date = st.date_input("Pilih Tanggal Mulai", pd.to_datetime("2020-01-01"))
end_date = st.date_input("Pilih Tanggal Akhir", pd.to_datetime("2023-12-31"))
time_step = st.slider("Time Step", 10, 100, 50)
epoch_option = st.selectbox("Jumlah Epoch", [12, 25, 50, 100])

# Download data
st.write(f"Mengambil data harga {asset} dari Yahoo Finance...")
df = yf.download(asset, start=start_date, end=end_date)

# Validasi data
if df.empty:
    st.error(f"⚠️ Data untuk {asset} tidak ditemukan pada rentang {start_date} s/d {end_date}. "
             f"Silakan pilih tanggal lain atau pastikan ticker benar.")
    st.stop()

# Persiapan data
closedf = df[['Close']]
scaler = MinMaxScaler(feature_range=(0, 1))
closedf_scaled = scaler.fit_transform(np.array(closedf).reshape(-1, 1))

train_size = int(len(closedf_scaled) * 0.7)
train_data, test_data = closedf_scaled[0:train_size, :], closedf_scaled[train_size:, :]

def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Validasi dataset
if len(X_train) == 0 or len(X_test) == 0:
    st.error("⚠️ Data tidak cukup untuk membuat dataset dengan time_step yang dipilih. "
             "Silakan kurangi time_step atau pilih rentang tanggal yang lebih panjang.")
    st.stop()

# Reshape input [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Model LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Training
with st.spinner("⏳ Melatih model..."):
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch_option, batch_size=32, verbose=0)

# Prediksi
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

# Metrik Evaluasi
train_rmse = math.sqrt(mean_squared_error(original_ytrain, train_predict))
test_rmse = math.sqrt(mean_squared_error(original_ytest, test_predict))
train_mae = mean_absolute_error(original_ytrain, train_predict)
test_mae = mean_absolute_error(original_ytest, test_predict)

st.subheader("📏 Metrik Evaluasi")
st.write(f"**Train RMSE:** {train_rmse:.2f}")
st.write(f"**Test RMSE:** {test_rmse:.2f}")
st.write(f"**Train MAE:** {train_mae:.2f}")
st.write(f"**Test MAE:** {test_mae:.2f}")

# Display Prediction Results
predict_dates = df.index[time_step+1:time_step+1+len(train_predict)+len(test_predict)]
result_df = pd.DataFrame({
    'Date': predict_dates,
    'Original_Close': np.concatenate([original_ytrain.flatten(), original_ytest.flatten()]),
    'Predicted_Close': np.concatenate([train_predict.flatten(), test_predict.flatten()])
})

# Plot hasil prediksi
st.write(f"### 🔮 Prediksi Harga {asset}")
fig = px.line(result_df, x='Date', y=['Original_Close', 'Predicted_Close'],
              labels={'value': 'Harga', 'Date': 'Tanggal'})
st.plotly_chart(fig)

# Tampilkan DataFrame
st.write("### 📊 Hasil Prediksi")
st.write(result_df)
