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

ALPHA_VANTAGE_API_KEY = "45G1G2AY7W8KD3S5"

# Title
st.title("📈 Prediksi Harga Cryptocurrency dengan LSTM")
st.write("Aplikasi ini memprediksi harga penutupan cryptocurrency menggunakan model LSTM (data Alpha Vantage).")

# Valid options
valid_time_steps = [25, 50, 75, 100]
valid_epochs = [12, 25, 50, 100]
default_time_step = 100
default_epoch = 25

# Session state
if 'model_ran' not in st.session_state:
    st.session_state.model_ran = False

# Input settings
col1, col2 = st.columns(2)
with col1:
    time_step = st.radio("⏳ Time Step (window)", options=valid_time_steps, index=valid_time_steps.index(default_time_step))
with col2:
    epoch_option = st.radio("🔄 Jumlah Epoch", options=valid_epochs, index=valid_epochs.index(default_epoch))

# Date selection
start_date = st.date_input("📅 Tanggal Mulai", pd.to_datetime("2020-01-01"))
end_date = st.date_input("📅 Tanggal Akhir", pd.to_datetime("2024-01-01"))

# Asset selection (hanya 2 aset)
asset_name_display = st.radio("💰 Pilih Aset", options=['BITCOIN', 'ETHEREUM'], index=0)

# Validasi Input
is_valid = (start_date < end_date)

@st.cache_data(ttl=60*60, show_spinner=False)
def fetch_crypto_data(symbol: str, market: str, start_date, end_date):
    """Fetch cryptocurrency daily data from Alpha Vantage."""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "DIGITAL_CURRENCY_DAILY",
        "symbol": symbol,
        "market": market,
        "apikey": ALPHA_VANTAGE_API_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        return {"error": f"Gagal terhubung ke Alpha Vantage: {exc}", "df": pd.DataFrame()}

    data = response.json()
    time_series_key = "Time Series (Digital Currency Daily)"

    if time_series_key not in data:
        message = data.get("Note") or data.get("Error Message") or "Data tidak tersedia."
        return {"error": f"Alpha Vantage tidak mengembalikan data: {message}", "df": pd.DataFrame()}

    time_series = data[time_series_key]
    df = pd.DataFrame.from_dict(time_series, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.rename(
        columns={
            "1a. open (USD)": "Open",
            "2a. high (USD)": "High",
            "3a. low (USD)": "Low",
            "4a. close (USD)": "Close",
            "5. volume": "Volume",
            "6. market cap (USD)": "Market Cap",
        }
    )

    numeric_columns = ["Open", "High", "Low", "Close", "Volume", "Market Cap"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_index()

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]

    return {"error": None, "df": df.reset_index().rename(columns={"index": "Date"})}

# Run Prediction Button
if st.button("🚀 Jalankan Prediksi", disabled=not is_valid):
    # Mapping assets ke simbol Alpha Vantage
    asset_mapping = {
        'BITCOIN': {'symbol': 'BTC', 'market': 'USD', 'display': 'BTC/USD'},
        'ETHEREUM': {'symbol': 'ETH', 'market': 'USD', 'display': 'ETH/USD'}
    }
    asset_config = asset_mapping[asset_name_display]

    # Fetch data
    with st.spinner(f"📥 Mengambil data harga {asset_name_display} ({asset_config['display']}) dari Alpha Vantage..."):
        result = fetch_crypto_data(
            symbol=asset_config['symbol'],
            market=asset_config['market'],
            start_date=start_date,
            end_date=end_date,
        )
    if result["error"]:
        st.error("❌ " + result["error"])
        st.session_state.model_ran = False
        st.stop()

    df = result["df"]
    if df.empty:
        st.warning("⚠️ Data tidak tersedia untuk rentang tanggal yang dipilih.")
        st.session_state.model_ran = False
        st.stop()

    if "Close" not in df.columns or df["Close"].isna().all():
        st.error("❌ Kolom 'Close' tidak ditemukan/bernilai NaN.")
        st.stop()

    if len(df) <= time_step + 1:
        st.warning("⚠️ Data tidak cukup untuk pelatihan dengan konfigurasi time step yang dipilih. Perluas rentang tanggal atau kecilkan time step.")
        st.session_state.model_ran = False
        st.stop()

    # Plot harga asli
    st.write(f"### 📊 Histori Harga Penutupan {asset_name_display}")
    fig = px.line(df, x='Date', y='Close', title=f'Histori Harga {asset_name_display}')
    fig.update_traces(line=dict(width=2))
    st.plotly_chart(fig, use_container_width=True)

    # Preprocessing
    closedf = df[['Close']].copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    closedf_scaled = scaler.fit_transform(np.array(closedf).reshape(-1, 1))

    # Split data (90% train, 10% test)
    training_size = int(len(closedf_scaled) * 0.90)
    train_data = closedf_scaled[:training_size]
    test_data = closedf_scaled[training_size:]

    # Function to create dataset (X sequences & y target)
    def create_dataset(dataset, time_step_local=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step_local - 1):
            a = dataset[i:(i + time_step_local), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step_local, 0])
        return np.array(dataX), np.array(dataY)

    # Buat sequence
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    if len(X_train) == 0 or len(X_test) == 0:
        st.error("❌ Data tidak cukup untuk membuat sequence. Coba kecilkan time_step atau perluas rentang tanggal.")
        st.stop()

    # Reshape ke [samples, time_step, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Model LSTM
    with st.spinner("🧠 Melatih model LSTM..."):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        history = model.fit(X_train, y_train, epochs=epoch_option, batch_size=32, verbose=0)

    # Prediksi
    train_predict = model.predict(X_train, verbose=0)
    test_predict = model.predict(X_test, verbose=0)

    # Invers scaling
    train_predict_inv = scaler.inverse_transform(train_predict)
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
    test_predict_inv = scaler.inverse_transform(test_predict)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    original_inv = scaler.inverse_transform(closedf_scaled)

    # Metrik
    train_mse = mean_squared_error(y_train_inv, train_predict_inv)
    test_mse = mean_squared_error(y_test_inv, test_predict_inv)
    test_rmse = math.sqrt(test_mse)

    mcol1, mcol2 = st.columns(2)
    mcol1.metric("MSE (Train)", f"{train_mse:,.4f}")
    mcol2.metric("RMSE (Test)", f"{test_rmse:,.4f}")

    # Siapkan plot Actual vs Train/Test Predict
    look_back = time_step

    trainPlot = np.empty_like(closedf_scaled)
    trainPlot[:, :] = np.nan
    trainPlot[look_back:len(train_predict_inv)+look_back, :] = train_predict_inv

    testPlot = np.empty_like(closedf_scaled)
    testPlot[:, :] = np.nan
    test_start = len(train_predict_inv) + (look_back * 2) + 1
    test_end = len(closedf_scaled) - 1
    # pastikan panjang cocok
    span = min(len(test_predict_inv), max(0, test_end - test_start + 1))
    if span > 0:
        testPlot[test_start:test_start+span, :] = test_predict_inv[:span]

    plot_df = pd.DataFrame({
        "Date": df["Date"].values,
        "Actual": original_inv.ravel(),
        "Train Predict": trainPlot.ravel(),
        "Test Predict": testPlot.ravel()
    })

    fig2 = px.line(plot_df, x="Date", y=["Actual", "Train Predict", "Test Predict"],
                   title=f"{asset_name_display} — Actual vs Train/Test Predict",
                   labels={"value": "Price (USD)", "variable": "Series"})
    fig2.update_traces(line=dict(width=2))
    st.plotly_chart(fig2, use_container_width=True)

    st.success("✅ Selesai! Model berhasil dilatih dan divisualisasikan.")
    st.caption("Catatan: Free tier Alpha Vantage memiliki rate limit (±5 request/menit). Jika muncul pesan Note/limit, coba tunggu sebentar lalu jalankan lagi.")
else:
    if not is_valid:
        st.info("Tanggal mulai harus sebelum tanggal akhir.")
    else:
        st.info("Atur parameter di atas, lalu klik **🚀 Jalankan Prediksi**.")
