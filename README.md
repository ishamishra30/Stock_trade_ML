# Stock_trade_ML

This project predicts short-term stock price direction (UP / DOWN) using:

1. Historical stock price data (Tesla – TSLA)

2. Technical Indicators (RSI & CCI)

3. Machine Learning model (Neural Network – MLPClassifier)

4. Strategy backtesting using predicted signals

🔹 Project Objective

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

🔹  Stock Data Download
df = yf.download('TSLA', start='2016-01-01', end=date.today())

✔ Downloads Tesla (TSLA) stock data from 2016 to today
✔ Columns include:

Open, High, Low, Close, Volume

| Term   | Meaning                  |
| ------ | ------------------------ |
| Open   | Price at market open     |
| High   | Highest price of the day |
| Low    | Lowest price of the day  |
| Close  | Price at market close    |
| Volume | Number of shares traded  |

🔹 4. Technical Indicators Used
✅ RSI – Relative Strength Index

RSI measures momentum and shows whether a stock is:

Overbought (RSI > 70 → price may fall)

Oversold (RSI < 30 → price may rise)

📌 Formula idea:
RSI compares recent gains vs recent losses

You used:

RSI(2)

RSI(7)

RSI(14)

✔ Smaller window = more sensitive
✔ Larger window = smoother trend

✅ CCI – Commodity Channel Index

CCI measures how far price has moved from its average price

| CCI Value  | Meaning               |
| ---------- | --------------------- |
| Above +100 | Strong upward trend   |
| Below -100 | Strong downward trend |
| Near 0     | Sideways market       |

You used:

CCI(30)

CCI(50)

CCI(100)

✔ Large window = long-term trend
✔ Small window = short-term momentum

🔹 Data Labeling (Target Variable)
df['LABEL'] = np.where(df['Open'].shift(-2) > df['Open'].shift(-1), "1", "0")

🎯 Meaning of LABEL:

| Label | Meaning         |
| ----- | --------------- |
| "1"   | Price went UP   |
| "0"   | Price went DOWN |

🔹 6. Feature Selection
x = df[df.columns[6:-1]].values
y = df['LABEL'].values


Features include:

RSI(2), RSI(7), RSI(14)

CCI(30), CCI(50), CCI(100)

Target:

LABEL (UP / DOWN)

🔹 Machine Learning Model
mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu')


✔ Neural Network with 3 hidden layers
✔ Learns non-linear market patterns
✔ Trained using historical indicator values

🔹 Model Evaluation Output
classification_report(y_test, predict_test)

Output Metrics Explained:

| Metric    | Meaning                             |
| --------- | ----------------------------------- |
| Precision | How many predicted UPs were correct |
| Recall    | How many actual UPs were caught     |
| F1-score  | Balance between Precision & Recall  |
| Accuracy  | Overall correctness                 |

📌 You get:

Train Accuracy

Test Accuracy

Confusion Matrix

This tells you how good your model is at predicting market direction.

🔹 Trading Strategy Simulation
df['Strategy Returns'] = np.where(df['Prediction']=="1", df['Open'].shift(-2) - df['Open'].shift(-1), 0)


📌 Strategy logic:

| Condition      | Action   |
| -------------- | -------- |
| Prediction = 1 | Buy      |
| Prediction = 0 | No trade |

✔ Calculates profit/loss
✔ Cumulative returns plotted

📈 This simulates how profitable your ML-based strategy would be.

🔹 10. Final Output

🎯 Final Output Meaning:

Your system predicts:

📢 Today's stock direction: UP or DOWN

This is your ML-based market signal.

🔹 Project Flow (High Level)
Download Stock Data
        ↓
Calculate RSI & CCI
        ↓
Create Labels (UP/DOWN)
        ↓
Train Neural Network
        ↓
Evaluate Accuracy
        ↓
Simulate Trading Strategy
        ↓
Predict Today’s Market Direction

| Term               | Meaning                             |
| ------------------ | ----------------------------------- |
| Indicator          | Mathematical formula based on price |
| Technical Analysis | Studying charts & indicators        |
| Buy Signal         | Suggests price may rise             |
| Sell Signal        | Suggests price may fall             |
| Overbought         | Too many buyers (possible drop)     |
| Oversold           | Too many sellers (possible rise)    |
| Backtesting        | Testing strategy on past data       |
| Strategy Returns   | Profit or loss from trades          |

using steamlit we can make better interface too!!

<img width="1920" height="1080" alt="Screenshot (36)" src="https://github.com/user-attachments/assets/2b8df7d3-0b4c-46bc-b33b-01e37892e904" />

<img width="1920" height="1080" alt="Screenshot (37)" src="https://github.com/user-attachments/assets/9ae354b4-4c28-4d3f-972f-c7215e01a933" />
