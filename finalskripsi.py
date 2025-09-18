 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/finalskripsi.py b/finalskripsi.py
index a5e786f05f7fb25db2395b9b512c301d21a61dfb..58bf3689929cd7555ed3931a7994e8021a6228dc 100644
--- a/finalskripsi.py
+++ b/finalskripsi.py
@@ -1,81 +1,154 @@
 import streamlit as st
-import yfinance as yf
+import requests
 import tensorflow as tf
 import numpy as np
 import pandas as pd
 import plotly.express as px
 import math
 from sklearn.preprocessing import MinMaxScaler
 from sklearn.metrics import mean_squared_error
 from tensorflow.keras.models import Sequential
 from tensorflow.keras.layers import Dense, LSTM
 
+ALPHA_VANTAGE_API_KEY = "45G1G2AY7W8KD3S5"
+
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
+def fetch_crypto_data(symbol: str, market: str, start_date, end_date):
+    """Fetch cryptocurrency daily data from Alpha Vantage."""
+    url = "https://www.alphavantage.co/query"
+    params = {
+        "function": "DIGITAL_CURRENCY_DAILY",
+        "symbol": symbol,
+        "market": market,
+        "apikey": ALPHA_VANTAGE_API_KEY,
+    }
+
+    try:
+        response = requests.get(url, params=params, timeout=30)
+        response.raise_for_status()
+    except requests.RequestException as exc:
+        st.error(f"❌ Gagal terhubung ke Alpha Vantage: {exc}")
+        return pd.DataFrame()
+
+    data = response.json()
+    time_series_key = "Time Series (Digital Currency Daily)"
+
+    if time_series_key not in data:
+        message = data.get("Note") or data.get("Error Message") or "Data tidak tersedia."
+        st.error(f"❌ Alpha Vantage tidak mengembalikan data: {message}")
+        return pd.DataFrame()
+
+    time_series = data[time_series_key]
+    df = pd.DataFrame.from_dict(time_series, orient="index")
+    df.index = pd.to_datetime(df.index)
+    df = df.rename(
+        columns={
+            "1a. open (USD)": "Open",
+            "2a. high (USD)": "High",
+            "3a. low (USD)": "Low",
+            "4a. close (USD)": "Close",
+            "5. volume": "Volume",
+            "6. market cap (USD)": "Market Cap",
+        }
+    )
+
+    numeric_columns = ["Open", "High", "Low", "Close", "Volume", "Market Cap"]
+    for column in numeric_columns:
+        if column in df.columns:
+            df[column] = pd.to_numeric(df[column], errors="coerce")
+
+    df = df.sort_index()
+
+    start_ts = pd.to_datetime(start_date)
+    end_ts = pd.to_datetime(end_date)
+    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
+
+    return df.reset_index().rename(columns={"index": "Date"})
+
+
 if st.button("🚀 Jalankan Prediksi", disabled=not is_valid):
-    
+
     # Mapping assets
-    asset_mapping = {'BITCOIN': 'BTC-USD', 'ETHEREUM': 'ETH-USD'}
-    asset = asset_mapping[asset_name_display]
+    asset_mapping = {
+        'BITCOIN': {'symbol': 'BTC', 'market': 'USD', 'display': 'BTC/USD'},
+        'ETHEREUM': {'symbol': 'ETH', 'market': 'USD', 'display': 'ETH/USD'}
+    }
+    asset_config = asset_mapping[asset_name_display]
 
     # Fetch data
-    st.write(f"📥 Mengambil data harga {asset_name_display} ({asset}) dari Yahoo Finance...")
-    df = yf.download(asset, start=start_date, end=end_date)
-    df = df.reset_index()
-    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
+    st.write(
+        f"📥 Mengambil data harga {asset_name_display} ({asset_config['display']}) dari Alpha Vantage..."
+    )
+    df = fetch_crypto_data(
+        symbol=asset_config['symbol'],
+        market=asset_config['market'],
+        start_date=start_date,
+        end_date=end_date,
+    )
+
+    if df.empty:
+        st.warning("⚠️ Data tidak tersedia untuk rentang tanggal yang dipilih.")
+        st.session_state.model_ran = False
+        st.stop()
+
+    if len(df) <= time_step + 1:
+        st.warning("⚠️ Data tidak cukup untuk melakukan pelatihan dengan konfigurasi time step yang dipilih.")
+        st.session_state.model_ran = False
+        st.stop()
 
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
 
EOF
)
