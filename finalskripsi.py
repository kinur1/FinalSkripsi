import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from io import StringIO
from datetime import datetime, date

st.set_page_config(page_title="Alpha Vantage Crypto Viewer", layout="wide", page_icon="📈")
st.title("📊 Cryptocurrency Data Viewer (Alpha Vantage)")

# === CONFIG ===
API_KEY = "45G1G2AY7W8KD3S5"
BASE_URL = "https://www.alphavantage.co/query"

# === Mapping asset → simbol Alpha Vantage ===
asset_mapping = {
    "Bitcoin": "BTC",
    "Ethereum": "ETH",
    "Binance Coin": "BNB",
    "Cardano": "ADA",
    "Ripple": "XRP",
    "Solana": "SOL",
    "Dogecoin": "DOGE"
}

# === Pilih aset ===
asset_name = st.selectbox("Pilih aset kripto:", list(asset_mapping.keys()))
ticker = asset_mapping[asset_name]

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Tanggal mulai", date.today().replace(year=date.today().year - 1))
with col2:
    end_date = st.date_input("Tanggal akhir", date.today())

if start_date >= end_date:
    st.error("Tanggal mulai harus sebelum tanggal akhir.")
    st.stop()

# === caching fetch supaya tidak berulang-ulang ===
@st.cache_data(ttl=60*60)
def fetch_alpha_vantage_symbol(symbol: str, apikey: str):
    params = {
        "function": "DIGITAL_CURRENCY_DAILY",
        "symbol": symbol,
        "market": "USD",
        "apikey": apikey
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    return r.status_code, r.json()

def pick_value_from_values(values: dict, field: str):
    keys = [k for k in values.keys() if field in k.lower()]
    usd_keys = [k for k in keys if "(usd)" in k.lower()]
    use_key = usd_keys[0] if usd_keys else (keys[0] if keys else None)
    if use_key is None:
        return 0.0
    try:
        return float(values.get(use_key, 0))
    except Exception:
        try:
            return float(str(values.get(use_key, "0")).replace(",", ""))
        except Exception:
            return 0.0

# === main ===
with st.spinner(f"Mengambil data {asset_name} ({ticker}) dari Alpha Vantage..."):
    status, payload = fetch_alpha_vantage_symbol(ticker, API_KEY)

if status != 200:
    st.error(f"Gagal memanggil Alpha Vantage (HTTP {status}) untuk {ticker}.")
    st.stop()

if "Note" in payload:
    st.error(f"Rate limit: {payload.get('Note')}")
    st.stop()

if "Error Message" in payload:
    st.error(f"Error dari Alpha Vantage: {payload.get('Error Message')}")
    st.stop()

if "Time Series (Digital Currency Daily)" not in payload:
    st.error(f"Tidak ada data historis untuk {ticker}.")
    st.stop()

# Parsing data
ts = payload["Time Series (Digital Currency Daily)"]
records = []
for dt_str, values in ts.items():
    records.append({
        "Date": dt_str,
        "Open": pick_value_from_values(values, "open"),
        "High": pick_value_from_values(values, "high"),
        "Low": pick_value_from_values(values, "low"),
        "Close": pick_value_from_values(values, "close"),
        "Volume": pick_value_from_values(values, "volume")
    })

df = pd.DataFrame(records)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)
mask = (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
df = df.loc[mask].reset_index(drop=True)

if df.empty:
    st.warning(f"Tidak ada data {asset_name} ({ticker}) untuk rentang {start_date} s/d {end_date}.")
    st.stop()

# tampilkan
st.subheader(f"{asset_name} ({ticker}) — data {start_date} s/d {end_date}")
st.dataframe(df.tail(20))

# chart harga penutupan
if "Close" in df.columns and df["Close"].notna().any():
    fig = px.line(df, x="Date", y="Close", title=f"{asset_name} ({ticker}) Closing Price (USD)",
                  labels={"Close": "Close (USD)"}, template="plotly_dark")
    fig.update_traces(line=dict(width=2.5))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning(f"Tidak ada kolom Close untuk {ticker}.")

# download CSV
csv_buf = StringIO()
df.to_csv(csv_buf, index=False)
st.download_button(f"⬇️ Download {ticker} CSV", data=csv_buf.getvalue(), file_name=f"{ticker}_data.csv", mime="text/csv")
