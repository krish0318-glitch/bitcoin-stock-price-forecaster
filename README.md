# Bitcoin & Stock Price Forecaster

An LSTM-based time-series forecasting project for Bitcoin and stocks.

## Tech stack
- Python
- Streamlit
- TensorFlow / Keras
- LSTM
- yfinance
- Pandas / NumPy
- Scikit-learn
- Plotly

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

This project is prepared for Streamlit Community Cloud.

1. Create a GitHub repository.
2. Upload `app.py` and `requirements.txt`.
3. Go to Streamlit Community Cloud.
4. Select the GitHub repository and `app.py` as the entrypoint.
5. Deploy.

The app downloads historical data dynamically from Yahoo Finance, so the CSV files from the original local project are not required for deployment.
