# LSTM-Stock-price-and-volume-predictor
An advanced AI-powered stock analytics platform that blends deep learning (LSTM), interactive analytical visualization, 7-day forecasts, and trend insights predictive intelligence,  to forecast market trends and empower smarter financial decisions.


├── app.py                 # Streamlit dashboard
├── KAYNES.csv             # Historical stock data of the company Kaynes Technology India Ltd
├── stock_prediction.h5    # Trained LSTM model
├── scalers.pkl            # Saved scalers
├── requirements.txt       # Dependencies
└── README.md              # Project documentation

Model Overview:

Model Type: Multi output regression based LSTM (Recurrent Neural Network)
Input Features: Closing Price, Volume, Moving Averages (MA10, MA20), Exponential Moving Average (EMA10), Returns & Volume Change
Output: Future Closing Price and Future Volume for future 7 days

The model captures time-series dependencies to forecast short-term price movements.

Features of the platform

- Interactive Dashboard with OHLC & Volume visualization
- 7-Day Future Price Prediction using LSTM
- Trend Classification (Bullish / Bearish / Consolidating)
- Dark/Light Mode Toggle
- Sidebar Forecast Table (Next 7 days price , volume, trend and ROI probability)
- Performance Metrics (Returns, High, Low, Volatility)
- Reset & Dynamic Controls

Trend Detection Logic

The application analyzes predicted prices to classify market direction:

- Bullish: Price expected to rise (> +2%)
- Bearish: Price expected to fall (< -2%)
- Consolidating: Sideways movement (-2% to +2%)

Tech Stack:

Python
TensorFlow / Keras (LSTM)
Pandas, NumPy
Plotly (Interactive Charts)
Streamlit (Web App

Future Improvements:

- We can Add technical indicators (RSI, MACD)
- Improve model with additional features (lag, sentiment)
- Add buy/sell signals
- Deploy with live data APIs for real time predictions for scalping and intraday trading decisions

Disclaimer ⚠️:

This project is for educational and research purposes only.
The predictions are generated using a machine learning model and should not be considered financial advice.
