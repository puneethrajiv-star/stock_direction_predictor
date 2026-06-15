** Stock Market Direction Prediction Using LightGBM

A machine learning project that uses Python and LightGBM to predict whether a stock's closing price will go up or down tomorrow.

-> Project Overview
This project frames stock prediction as a **Binary Classification** task:
* **1 (True):** Tomorrow's closing price is higher than today's.
* **0 (False):** Tomorrow's closing price is lower or equal to today's.

Instead of predicting the exact price, the model focuses strictly on predicting the market's direction.

-> Features Used
To make the model stable across different years, the code transforms raw stock data into relative market signals:
1. **Returns:** The daily percentage change in the stock's closing price.
2. **Volume Ratio:** Today's trading volume divided by its 20-day moving average (shows if buying activity is higher or lower than usual).

-> How It Works
* **Data Source:** Fetches historical data for Apple (`AAPL`) using the `yfinance` library.
* **Train/Test Split:** Chronological split to simulate real-world testing. Trains on data from 2023–2024 and tests out-of-sample on data from 2025–2026.
* **Data Scaling:** Uses `RobustScaler` to handle outliers in financial data.
* **Model:** Trains a `LightGBM` classifier using balanced class weights.
* **Visualization:** Generates a horizontal bar chart showing the feature importance of our inputs.

## Requirements
* Python 3.x
* yfinance
* lightgbm
* scikit-learn
* pandas
* matplotlib
