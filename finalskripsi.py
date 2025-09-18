import os
import math
from datetime import date
from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# ============== PAGE CONFIG ==============
st.set_page_config(page_title="BTC & ETH LSTM (Alpha Vantage)", layout="wide", page_icon="📈")
st.title("📈 Prediksi BTC & ETH dengan LSTM (Alpha Vantage)")

# ============== CONSTANTS ==============
DEFAULT_API_KEY = "45G1G2AY7W8KD3S5"  # ganti jika perlu
BASE_URL = "https://www.alphavantage.co/query"
ASSETS = ["BTC", "ETH"]  # hanya 2 aset sesuai permintaan

VALID_TIME_STEPS = [25, 50, 75, 100]
VALID_EPOCHS = [12, 25, 50, 100]

# ============== SIDEBAR CONTROLS ==============
st.sidebar.header("⚙️ Pengaturan")
api_key = st.sidebar.text_input("Alpha Vantage API Key", value=DEFAULT_API_KEY, type="password")
col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    start_date = st.date_input("Tanggal mulai", date.today().replace(year=date.today().year - 3))
with col_s2:
    end_date = st.date_input("Tanggal akhir", date.today())

time_step = st.sidebar.selectbox("⏳ Time Step (window)", VALID_TIME_STEPS, index=VALID_TIME_STEPS.index(50))
epochs = st.sidebar.selectbox("🔁 Epochs", VALID_EPOCHS, index=VALID_EPOCHS.index(25))
forecast_days = st.sidebar.number_input("🔮 Hari forecast ke depan", min_value=1, max_value=60, value=14, step=1)
test_size_ratio = st.sidebar.slider("📊 Proporsi data test", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

run_button = st.sidebar.button("🚀 Jalankan Training & Prediksi")

if start_date >= end_date:
    st.error("Tanggal mulai harus sebelum tanggal akhir.")
    st.stop()

if not api_key:
    st.error("Masukkan API Key Alpha Vantage terlebih dahulu.")
    st.stop()

st.caption("Catatan: Free tier Alpha Vantage ada rate limit (±5 request/menit).")

# ============== HELPERS ==============
def pick_value_from_values(values: dict, field: str):
    """
    Memilih nilai berdasarkan nama kolom yang mengandung field (open/high/low/close/volume)
    dengan preferensi '(USD)' jika ada.
    """
    keys = [k for k in values.keys() if field in k.lower()]
    usd_keys = [k for k in keys if '(usd)' in k.lower()]
    use_key = usd_keys[0] if usd_keys else (keys[0] if keys else None)
    if use_key is None:
        # fallback volume
        vol_keys = [k for k in values.keys() if 'volume' in k.lower()]
        use_key = vol_keys[0] if vol_keys else None
    if use_key is None:
        return 0.0
    try:
        return float(values.get(use_key, 0))
    except Exception:
        try:
            return float(str(values.get(use_key, "0")).replace(",", ""))
        except Exception:
            return 0.0

@st.cache_data(ttl=60 * 60, show_spinner=False)  # cache 1 jam
def fetch_alpha_vantage_symbol(symbol: str, apikey: str):
    params = {
        "function": "DIGITAL_CURRENCY_DAILY",
        "symbol": symbol,
        "market": "USD",
        "apikey": apikey,
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    status = r.status_code
    try:
        payload = r.json()
    except Exception:
        payload = {}
    return status, payload

def parse_timeseries(payload: dict) -> pd.DataFrame:
    if "Time Series (Digital Currency Daily)" not in payload:
        return pd.DataFrame()
    ts = payload["Time Series (Digital Currency Daily)"]
    records = []
    for dt_str, values in ts.items():
        c = pick_value_from_values(values, "close")
        records.append({"Date": dt_str, "Close": c})
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def filter_by_date(df: pd.DataFrame, start_date_: date, end_date_: date) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (df["Date"].dt.date >= start_date_) & (df["Date"].dt.date <= end_date_)
    return df.loc[mask].reset_index(drop=True)

def make_sequences(series: np.ndarray, window: int):
    """
    series: shape (N, 1)
    return X: (N-window, window, 1), y: (N-window, 1)
    """
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window:i, 0])
        y.append(series[i, 0])
    X = np.array(X)
    y = np.array(y)
    return X.reshape((X.shape[0], X.shape[1], 1)), y

def build_lstm_model(window: int):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def train_and_predict(df: pd.DataFrame, window: int, epochs: int, test_ratio: float, future_days: int):
    """
    df: harus punya kolom ['Date','Close'] terurut
    """
    # scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = df[["Close"]].values.astype(float)
    scaled = scaler.fit_transform(values)

    # split train/test
    split_idx = int(len(scaled) * (1 - test_ratio))
    train_data = scaled[:split_idx]
    test_data = scaled[split_idx - window:]  # overlap window untuk sequence awal test

    # sequences
    X_train, y_train = make_sequences(train_data, window)
    X_test, y_test = make_sequences(test_data, window)

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Data tidak cukup untuk membuat sequence. Coba kecilkan time_step atau perluas rentang tanggal.")

    # model
    model = build_lstm_model(window)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)

    # predict test
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1))

    # forecast future
    last_window = scaled[-window:].reshape(1, window, 1).copy()
    future_scaled = []
    for _ in range(future_days):
        next_scaled = model.predict(last_window, verbose=0)[0][0]
        future_scaled.append([next_scaled])
        # slide window
        new_window = np.append(last_window[0, 1:, 0], next_scaled).reshape(1, window, 1)
        last_window = new_window

    future_forecast = scaler.inverse_transform(np.array(future_scaled))

    return {
        "model": model,
        "history": history,
        "y_true": y_true.ravel(),
        "y_pred": y_pred.ravel(),
        "future_forecast": future_forecast.ravel(),
        "split_index": split_idx,
    }

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    mse = float(np.mean((y_pred - y_true) ** 2))
    rmse = float(math.sqrt(mse))
    return mse, rmse

def plot_actual_vs_predicted(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, split_idx: int, asset: str):
    # y_true/predicted mulai dari split_idx (test bagian) tetapi minus window offset internal
    test_dates = df["Date"].iloc[split_idx:].reset_index(drop=True)
    # Sesuaikan panjangnya (y_true & y_pred punya panjang yang sama)
    test_dates = test_dates.iloc[-len(y_true):].reset_index(drop=True)

    plot_df = pd.DataFrame({
        "Date": pd.to_datetime(test_dates),
        "Actual": y_true,
        "Predicted": y_pred
    })
    fig = px.line(plot_df, x="Date", y=["Actual", "Predicted"],
                  title=f"{asset} — Actual vs Predicted (Test Set)",
                  labels={"value": "Price (USD)", "variable": "Series"})
    fig.update_traces(line=dict(width=2))
    return fig

def plot_forecast(df: pd.DataFrame, future_forecast: np.ndarray, future_days: int, asset: str):
    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq="D")

    hist_df = df[["Date", "Close"]].copy()
    hist_df["Type"] = "History"
    fut_df = pd.DataFrame({"Date": future_dates, "Close": future_forecast, "Type": "Forecast"})
    plot_df = pd.concat([hist_df.tail(120), fut_df], ignore_index=True)

    fig = px.line(plot_df, x="Date", y="Close", color="Type",
                  title=f"{asset} — Forecast {future_days} Hari ke Depan",
                  labels={"Close": "Price (USD)"})
    fig.update_traces(line=dict(width=2))
    return fig

# ============== MAIN CONTENT ==============
tabs = st.tabs([f"💰 {asset}" for asset in ASSETS])

if run_button:
    for asset, tab in zip(ASSETS, tabs):
        with tab:
            st.subheader(f"{asset}-USD — Rentang {start_date} s/d {end_date}")

            with st.spinner(f"Mengambil data {asset} dari Alpha Vantage..."):
                status, payload = fetch_alpha_vantage_symbol(asset, api_key)

            # error handling
            if status != 200:
                st.error(f"Gagal memanggil Alpha Vantage (HTTP {status}) untuk {asset}.")
                continue
            if "Note" in payload:
                st.error(f"Alpha Vantage Note (rate limit?): {payload.get('Note')}")
                continue
            if "Error Message" in payload:
                st.error(f"Alpha Vantage Error untuk {asset}: {payload.get('Error Message')}")
                continue

            df = parse_timeseries(payload)
            if df.empty:
                st.warning(f"Tidak ada data harian untuk {asset}.")
                continue

            df = filter_by_date(df, start_date, end_date)
            if df.empty:
                st.warning(f"Tidak ada data {asset} pada rentang tanggal {start_date} s/d {end_date}.")
                continue

            st.dataframe(df.tail(20), use_container_width=True)

            # garis harga historis
            fig_hist = px.line(df, x="Date", y="Close",
                               title=f"{asset} — Closing Price (USD)",
                               labels={"Close": "Close (USD)"})
            fig_hist.update_traces(line=dict(width=2))
            st.plotly_chart(fig_hist, use_container_width=True)

            # minimal data check
            min_needed = max(50, time_step + 50)  # rule of thumb: butuh data yang cukup
            if len(df) < (time_step + 10):
                st.error(f"Data {asset} terlalu sedikit ({len(df)} baris). Perlu ≥ {time_step + 10} baris. "
                         f"Coba kecilkan time_step atau perluas rentang tanggal.")
                continue

            try:
                with st.spinner(f"Melatih model LSTM untuk {asset}..."):
                    result = train_and_predict(
                        df=df,
                        window=time_step,
                        epochs=epochs,
                        test_ratio=test_size_ratio,
                        future_days=forecast_days
                    )
                mse, rmse = compute_metrics(result["y_true"], result["y_pred"])

                # Metrics
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("MSE (Test)", f"{mse:,.4f}")
                with m2:
                    st.metric("RMSE (Test)", f"{rmse:,.4f}")

                # Plot Actual vs Predicted
                fig_pred = plot_actual_vs_predicted(
                    df=df,
                    y_true=result["y_true"],
                    y_pred=result["y_pred"],
                    split_idx=result["split_index"],
                    asset=asset
                )
                st.plotly_chart(fig_pred, use_container_width=True)

                # Forecast plot
                fig_fc = plot_forecast(df, result["future_forecast"], forecast_days, asset)
                st.plotly_chart(fig_fc, use_container_width=True)

                # Download CSV (subset + pred)
                csv_buf = StringIO()
                out_df = df.copy()
                out_df["Date"] = out_df["Date"].dt.strftime("%Y-%m-%d")
                out_df.to_csv(csv_buf, index=False)
                st.download_button(
                    f"⬇️ Download {asset} (History) CSV",
                    data=csv_buf.getvalue(),
                    file_name=f"{asset}_history_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )

                # Prediksi test set (date + actual + predicted)
                pred_df = pd.DataFrame({
                    "Date": pd.to_datetime(df["Date"].iloc[result["split_index"]:]).reset_index(drop=True).iloc[-len(result["y_true"]):],
                    "Actual": result["y_true"],
                    "Predicted": result["y_pred"]
                })
                pred_buf = StringIO()
                pred_df.to_csv(pred_buf, index=False)
                st.download_button(
                    f"⬇️ Download {asset} (Test Predictions) CSV",
                    data=pred_buf.getvalue(),
                    file_name=f"{asset}_test_predictions.csv",
                    mime="text/csv"
                )

                # Forecast CSV
                fc_df = pd.DataFrame({
                    "Date": pd.date_range(start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days, freq="D"),
                    "Forecast": result["future_forecast"]
                })
                fc_buf = StringIO()
                fc_df.to_csv(fc_buf, index=False)
                st.download_button(
                    f"⬇️ Download {asset} (Future Forecast {forecast_days}d) CSV",
                    data=fc_buf.getvalue(),
                    file_name=f"{asset}_future_forecast_{forecast_days}d.csv",
                    mime="text/csv"
                )

            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                st.exception(e)
else:
    st.info("Pilih parameter di sidebar, lalu klik **🚀 Jalankan Training & Prediksi** untuk mulai.")
