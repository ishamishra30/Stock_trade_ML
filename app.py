import streamlit as st

st.title("📈 Stock Price Direction Prediction App")
st.write("Using RSI, CCI & Neural Network (MLP)")

try:
    st.write("Step 1: Importing libraries...")

    import pandas as pd
    import numpy as np
    import yfinance as yf
    from ta.momentum import RSIIndicator
    from ta.trend import CCIIndicator
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from datetime import date

    st.write("Step 2: Libraries imported ✅")

    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2016-01-01"))
    end_date = st.sidebar.date_input("End Date", value=date.today())

    st.write("Step 3: Downloading data...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    st.write("Downloaded shape:", df.shape)

    if df.empty:
        st.error("No data downloaded. Try AAPL, MSFT, TSLA")
        st.stop()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    st.write("Step 4: Calculating indicators...")

    df['RSI(2)'] = RSIIndicator(df['Close'], window=2).rsi()
    df['RSI(7)'] = RSIIndicator(df['Close'], window=7).rsi()
    df['RSI(14)'] = RSIIndicator(df['Close'], window=14).rsi()

    df['CCI(30)'] = CCIIndicator(df['High'], df['Low'], df['Close'], window=30).cci()
    df['CCI(50)'] = CCIIndicator(df['High'], df['Low'], df['Close'], window=50).cci()
    df['CCI(100)'] = CCIIndicator(df['High'], df['Low'], df['Close'], window=100).cci()

    df.dropna(inplace=True)
    st.write("Step 5: Indicators calculated ✅")

    st.write("Step 6: Creating labels...")
    df['Future_Return'] = df['Close'].shift(-1) / df['Close'] - 1
    df['LABEL'] = np.where(df['Future_Return'] > 0.002, "1", "0")  # 0.2% threshold
    df.dropna(inplace=True)

    X = df[['RSI(2)', 'RSI(7)', 'RSI(14)', 'CCI(30)', 'CCI(50)', 'CCI(100)']].values
    y = df['LABEL'].values

    st.write("Step 7: Training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
    model = MLPClassifier(hidden_layer_sizes=(8,8,8), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    st.write("Step 8: Model trained ✅")

    st.subheader("📊 Model Performance")
    st.code("Train Report:\n" + classification_report(y_train, model.predict(X_train)))
    st.code("Test Report:\n" + classification_report(y_test, model.predict(X_test)))

    latest_features = df[['RSI(2)', 'RSI(7)', 'RSI(14)', 'CCI(30)', 'CCI(50)', 'CCI(100)']].iloc[-1].values.reshape(1, -1)
    prediction = model.predict(latest_features)[0]

    st.subheader("📢 Today's Prediction")
    st.success("📈 Market Direction: UP" if prediction == "1" else "📉 Market Direction: DOWN")

except Exception as e:
    st.error("❌ Your app crashed with this error:")
    st.exception(e)
