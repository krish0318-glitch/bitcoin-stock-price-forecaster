import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(
    page_title="Bitcoin & Stock Price Forecaster",
    page_icon="📈",
    layout="wide"
)

NAME_MAP = {
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
}

@st.cache_data(ttl=3600)
def get_data(ticker):
    """Download historical daily Close prices with a fallback method."""
    last_error = None

    # Method 1: Ticker.history(period="max") is generally more robust on hosted servers.
    try:
        data = yf.Ticker(ticker).history(
            period="max",
            interval="1d",
            auto_adjust=False,
            actions=False,
            timeout=30,
        )

        if data is not None and not data.empty and "Close" in data.columns:
            data = data.reset_index()
            data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
            data = data.dropna(subset=["Close"]).copy()
            if len(data) > 70:
                return data
    except Exception as exc:
        last_error = exc

    # Method 2: fallback to yf.download().
    try:
        data = yf.download(
            ticker,
            period="max",
            interval="1d",
            auto_adjust=False,
            progress=False,
            timeout=30,
        )

        if data is not None and not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                # For a single ticker, keep the OHLC level.
                if ticker in data.columns.get_level_values(-1):
                    data = data.xs(ticker, axis=1, level=-1)
                else:
                    data.columns = data.columns.get_level_values(0)

            if "Close" in data.columns:
                data = data.reset_index()
                data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
                data = data.dropna(subset=["Close"]).copy()
                if len(data) > 70:
                    return data
    except Exception as exc:
        last_error = exc

    if last_error:
        raise ValueError(
            f"Yahoo Finance could not return data for '{ticker}'. "
            "Check the symbol and try again."
        )

    return pd.DataFrame()

def build_sequences(close_prices, window_size=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices.reshape(-1, 1))

    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i, 0])
        y.append(scaled[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    train_size = int(len(X) * 0.8)
    return (
        X[:train_size], X[train_size:],
        y[:train_size], y[train_size:],
        scaler
    )

def train_and_predict(ticker, epochs, window_size):
    data = get_data(ticker)

    if data.empty:
        raise ValueError(
            f"No historical data was returned for '{ticker}'. "
            "Check the ticker symbol or try again in a moment."
        )

    if len(data) <= window_size + 10:
        raise ValueError(
            f"Only {len(data)} historical observations were returned for "
            f"'{ticker}', which is not enough for a {window_size}-day window."
        )

    close_prices = data["Close"].values.astype(float)
    X_train, X_test, y_train, y_test, scaler = build_sequences(
        close_prices, window_size
    )

    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(window_size, 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=32,
        verbose=0
    )

    predictions_scaled = model.predict(X_test, verbose=0)
    predictions = scaler.inverse_transform(predictions_scaled).flatten()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
    errors = actual - predictions

    return data, actual, predictions, errors, history.history["loss"], rmse

st.title("📈 Bitcoin & Stock Price Forecaster")
st.caption("LSTM-based time-series forecasting using Yahoo Finance historical data")

with st.sidebar:
    st.header("Model Settings")
    ticker = st.text_input(
        "Stock / Crypto Symbol",
        value="BTC-USD",
        help="Examples: BTC-USD, ETH-USD, AAPL, TSLA"
    ).strip().upper()
    epochs = st.slider("Training Epochs", 5, 20, 15)
    window_size = st.slider("Sliding Window", 30, 90, 60)

    st.info(
        "The model uses the previous 60 trading/daily observations "
        "to predict the next observation."
    )

run = st.button("🚀 Train & Forecast", type="primary", use_container_width=True)

if run:
    if not ticker:
        st.error("Please enter a stock or crypto symbol.")
        st.stop()

    with st.spinner(f"Downloading data and training LSTM for {ticker}..."):
        try:
            data, actual, predictions, errors, losses, rmse = train_and_predict(
                ticker, epochs, window_size
            )
        except Exception as e:
            st.error(f"Could not run the model: {e}")
            st.stop()

    display_name = NAME_MAP.get(ticker, ticker)

    c1, c2, c3 = st.columns(3)
    c1.metric("Asset", display_name)
    c2.metric("Latest Close", f"${data['Close'].iloc[-1]:,.2f}")
    c3.metric("RMSE", f"${rmse:,.2f}")

    st.success(f"Forecast completed for {display_name}.")

    st.subheader("Actual vs Predicted Price")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=actual, mode="lines", name="Actual Price"
    ))
    fig.add_trace(go.Scatter(
        y=predictions, mode="lines", name="Predicted Price"
    ))
    fig.update_layout(
        xaxis_title="Test Set Time",
        yaxis_title="Price",
        hovermode="x unified",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prediction Error")
    error_fig = go.Figure()
    error_fig.add_trace(go.Scatter(
        y=errors, mode="lines", name="Error"
    ))
    error_fig.update_layout(
        xaxis_title="Test Set Time",
        yaxis_title="Actual - Predicted",
        height=350
    )
    st.plotly_chart(error_fig, use_container_width=True)

    st.subheader("Training Loss")
    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(
        x=list(range(1, len(losses) + 1)),
        y=losses,
        mode="lines+markers",
        name="Training Loss"
    ))
    loss_fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Mean Squared Error Loss",
        height=350
    )
    st.plotly_chart(loss_fig, use_container_width=True)

    with st.expander("Model / Dataset Details"):
        st.write(f"**Data points:** {len(data):,}")
        st.write(f"**Training samples:** {int(len(actual) / 0.2 * 0.8):,} (approximately)")
        st.write(f"**Test samples:** {len(actual):,}")
        st.write(f"**Sliding window:** {window_size}")
        st.write(f"**Epochs:** {epochs}")
        st.write("**Model:** LSTM (50 units) + Dense output layer")
        st.write("**Optimizer:** Adam")
        st.write("**Loss:** Mean Squared Error")

else:
    st.markdown(
        """
        ### How it works
        1. Fetches historical stock/crypto data using **Yahoo Finance**.
        2. Uses the **Close** price for preprocessing.
        3. Scales prices with **MinMaxScaler**.
        4. Creates **60-day sliding-window sequences**.
        5. Splits the data into **80% training / 20% testing**.
        6. Trains an **LSTM neural network**.
        7. Compares actual and predicted prices and calculates **RMSE**.
        """
    )
