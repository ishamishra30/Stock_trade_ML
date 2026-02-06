# Stock_trade_ML

This project predicts short-term stock price direction (UP / DOWN) using:

1. Historical stock price data (Tesla – TSLA)

2. Technical Indicators (RSI & CCI)

3. Machine Learning model (Neural Network – MLPClassifier)

4. Strategy backtesting using predicted signals

🔹 1. Project Objective

   The goal of this project is to:

1. Analyze historical stock price data

2. Calculate technical indicators used in trading

3. Train a Machine Learning model to predict whether the stock price will go UP or DOWN

4. Evaluate the model’s performance

5. Simulate a simple trading strategy

6. Predict today's market direction

| Library           | Purpose                         |
| ----------------- | ------------------------------- |
| `pandas`          | Data manipulation               |
| `matplotlib`      | Data visualization              |
| `yfinance`        | Download stock market data      |
| `yahoofinancials` | Financial data handling         |
| `ta`              | Technical indicators (RSI, CCI) |
| `numpy`           | Numerical operations            |
| `scikit-learn`    | Machine Learning                |
| `MLPClassifier`   | Neural Network model            |

🔹 3. Stock Data Download
df = yf.download('TSLA', start='2016-01-01', end=date.today())
✔ Downloads Tesla (TSLA) stock data from 2016 to today
✔ Columns include:
Open
High
Low
Close
Volume


