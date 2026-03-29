# Imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow.keras.models import load_model
import joblib

st.set_page_config(page_title="Stock Dashboard", layout="wide")

# creating a title for the dashboard
st.title("📈 Stock Prediction Dashboard")

st.markdown("""
### 🏢 Kaynes Technology India Ltd  
**Ticker:** NSE: KAYNES
""")

# lading CSV, Model file and scaler file
df = pd.read_csv("KAYNES.csv")
model = load_model("stock_prediction.h5")
scalers = joblib.load("scalers.pkl")


# cleaning the datset
df.columns = df.columns.str.strip()
df = df.rename(columns={
    "Date": "date",
    "Open Price": "open",
    "High Price": "high",
    "Low Price": "low",
    "Close Price": "close",
    "Total Traded Quantity": "volume"
})

for col in ['open','high','low','close','volume']:
    df[col] = df[col].astype(str).str.replace(',', '')
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['date'] = pd.to_datetime(df['date'], format='%d-%b-%y')
df = df.sort_values('date')
df.set_index('date', inplace=True)
df = df.dropna()


#-----------------------------
# Features on "side-bar"
#-----------------------------

# ----- A stlyzed html format message to display disclaimer

st.sidebar.markdown("""
<div style="background-color:#ffe6e6; padding:10px; border-radius:10px;">
<b>⚠️ Disclaimer</b><br><br>
This app is for <b>educational purposes only</b>.<br>
Predictions are model-based and <b>not financial advice</b>.<br>
Always do your own research.
</div>
""", unsafe_allow_html=True)

# ----- creating a date range selector

st.sidebar.header(" 📅 Date Range Selector")

default_start = df.index.min()
default_end = df.index.max()

# Initialize session state
if "start_date_input" not in st.session_state:
    st.session_state.start_date_input = default_start

if "end_date_input" not in st.session_state:
    st.session_state.end_date_input = default_end

# Reset Button (updates widget keys directly)
if st.sidebar.button("🔄 Reset Date Range"):
    st.session_state.start_date_input = default_start
    st.session_state.end_date_input = default_end
    st.rerun()

# Date inputs bound to the same reset button keys
start_date = st.sidebar.date_input(
    "Start Date",
    key="start_date_input"
)

end_date = st.sidebar.date_input(
    "End Date",
    key="end_date_input"
)
# the range selected is fed to filtered_df
filtered_df = df.loc[start_date:end_date]

# METRICS - measuring the metric parameters using the data from range selector window

st.subheader(f"Metrics for the Date range selected")

if len(filtered_df) > 1:
    start_price = filtered_df['close'].iloc[0]
    end_price = filtered_df['close'].iloc[-1]

    returns = ((end_price - start_price) / start_price) * 100
    high = filtered_df['high'].max()
    low = filtered_df['low'].min()
    volatility = filtered_df['close'].pct_change().std() * 100
else:
    returns, high, low, volatility = 0, 0, 0, 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("📈 Return %", f"{returns:.2f}%")
c2.metric("🟢 High", f"{high:.2f}")
c3.metric("🔴 Low", f"{low:.2f}")
c4.metric("⚡ Volatility", f"{volatility:.2f}%")


# TREND ANALYSIS - # finding the trend using the data from range selector window


if len(filtered_df) > 1:
    start_price = filtered_df['close'].iloc[0]
    end_price = filtered_df['close'].iloc[-1]

    change_pct = ((end_price - start_price) / start_price) * 100

    if change_pct > 2:           # if percentage change is more than 2% Bullish
        trend = "🟢 Bullish"
    elif change_pct < -2:        # if percentage change is less than 2% Bearish
        trend = "🔴 Bearish"
    else:                        # if percentage change is between +/- 2% sideways
        trend = "🟡 Consolidating"
else:
    trend = "Not enough data"
    change_pct = 0

st.subheader(f"Trend for the Date range selected: {trend}")
st.write(f"Change: {round(change_pct, 2)}%")


# We will feature engineer as we did for training

df_feat = df[['close','volume']].copy()

df_feat['ma10'] = df_feat['close'].rolling(10).mean()
df_feat['ma20'] = df_feat['close'].rolling(20).mean()
df_feat['ema10'] = df_feat['close'].ewm(span=10).mean()
df_feat['returns'] = df_feat['close'].pct_change()
df_feat['vol_change'] = df_feat['volume'].pct_change()

df_feat = df_feat.dropna()

features = ['close','volume','ma10','ma20','ema10','returns','vol_change']


# Scaling  -  we use minmax scaler

scaled_cols = []
for col in features:
    scaled = scalers[col].transform(df_feat[[col]])
    scaled_cols.append(scaled)

scaled_data = np.hstack(scaled_cols)

# test predictions

lookback = 60
split_index = int(len(df_feat) * 0.8)

X_test = []
for i in range(split_index, len(scaled_data)):
    X_test.append(scaled_data[i-lookback:i])

X_test = np.array(X_test)

pred = model.predict(X_test, verbose=0)

pred_close = scalers['close'].inverse_transform(pred[:,0].reshape(-1,1)).flatten()
pred_volume = scalers['volume'].inverse_transform(pred[:,1].reshape(-1,1)).flatten()

pred_index = df_feat.index[split_index:]
pred_index = pred_index[:len(pred_close)]

pred_df = pd.DataFrame({
    'close': pred_close,
    'volume': pred_volume
}, index=pred_index)


# Future prediction same as the training model

future_steps = 7
future_data = df_feat.copy()
future_preds = []

for _ in range(future_steps):

    recent = future_data.iloc[-lookback:]

    scaled_window = []
    for col in features:
        scaled = scalers[col].transform(recent[[col]])
        scaled_window.append(scaled)

    scaled_window = np.hstack(scaled_window)

    pred = model.predict(
        scaled_window.reshape(1, lookback, len(features)),
        verbose=0
    )[0]

    pred_close = scalers['close'].inverse_transform([[pred[0]]])[0][0]
    pred_volume = scalers['volume'].inverse_transform([[pred[1]]])[0][0]

    next_date = future_data.index[-1] + pd.Timedelta(days=1)

    new_row = pd.DataFrame({
        'close': [pred_close],
        'volume': [pred_volume]
    }, index=[next_date])

    temp = pd.concat([future_data, new_row])

    temp['ma10'] = temp['close'].rolling(10).mean()
    temp['ma20'] = temp['close'].rolling(20).mean()
    temp['ema10'] = temp['close'].ewm(span=10).mean()
    temp['returns'] = temp['close'].pct_change()
    temp['vol_change'] = temp['volume'].pct_change()

    temp = temp.dropna()
    future_data = temp.copy()

    future_preds.append([pred_close, pred_volume])

future_preds = np.array(future_preds)

future_dates = pd.date_range(
    start=df_feat.index[-1] + pd.Timedelta(days=1),
    periods=future_steps
)

future_df = pd.DataFrame({
    "Closing Price": future_preds[:,0],
    "Volume": future_preds[:,1]
}, index=future_dates)


# ----- SIDEBAR TABLE

# Trend Classification - same logic is used when we classify the trend for date range selector

def classify_trend(future_df):
    start_price = future_df['Closing Price'].iloc[0]
    end_price = future_df['Closing Price'].iloc[-1]

    change_pct = ((end_price - start_price) / start_price) * 100

    if change_pct > 2:
        trend = "🟢 Bullish"
    elif change_pct < -2:
        trend = "🔴 Bearish"
    else:
        trend = "🟡 Consolidating"

    return trend, change_pct

trend, change = classify_trend(future_df)

st.sidebar.subheader("📊 Next 7-Days Trend Forecast ")

st.sidebar.metric(
    label=" Market Direction",
    value=trend,
    delta=f"{change:.2f}%"
)

st.sidebar.subheader("🔮 Next 7 Days Price and Volume Prediction")
st.sidebar.dataframe(future_df)

# THEME
theme = st.sidebar.toggle(" Chart Theme : 🌙 Dark Mode", value=False)
bg_color = "black" if theme else "white"
font_color = "white" if theme else "black"


fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])

# OHLC
fig.add_trace(go.Candlestick(
    x=filtered_df.index,
    open=filtered_df['open'],
    high=filtered_df['high'],
    low=filtered_df['low'],
    close=filtered_df['close'],
    name='OHLC'
), row=1, col=1)

# chart for Predicted data
fig.add_trace(go.Scatter(
    x=pred_df.index,
    y=pred_df['close'],
    name='Predicted',
    line=dict(color='blue', width=2)
), row=1, col=1)

# chart for future data point
fig.add_trace(go.Scatter(
    x=future_df.index,
    y=future_df['Closing Price'],
    name='Future',
    line=dict(color='#B8860B', dash='dash')
), row=1, col=1)

# Highlighting the predicted region
fig.add_vrect(
    x0=pred_df.index[0],
    x1=pred_df.index[-1],
    fillcolor="rgba(0, 0, 255, 0.1)",
    line_width=0
)

# Highlighting the future region
fig.add_vrect(
    x0=future_df.index[0],
    x1=future_df.index[-1],
    fillcolor="rgba(255, 165, 0, 0.15)",
    line_width=0
)

# Volume profile stack
colors = ['green']
for i in range(1, len(filtered_df)):
    colors.append(
        'green' if filtered_df['close'].iloc[i] >= filtered_df['close'].iloc[i-1]
        else 'red'
    )

fig.add_trace(go.Bar(
    x=filtered_df.index,
    y=filtered_df['volume'],
    marker_color=colors,
    name='Volume'
), row=2, col=1)

fig.update_layout(
    height=700,
    hovermode="x unified",
    plot_bgcolor=bg_color,
    paper_bgcolor=bg_color,
    font=dict(color=font_color)
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Model")
st.write("LSTM")
st.subheader("Data Procurement")
st.write(" The dataset used is the NSE listed company :  Kaynes Technology India Ltd."
         " The dataset procured for this company is from 28th November 2022 - 27th Feb 2026 ")
st.write("data was procured from NSE website : https://www.nseindia.com/report-detail/eq_security")

