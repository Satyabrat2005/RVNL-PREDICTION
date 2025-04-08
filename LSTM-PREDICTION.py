pip install yfinance numpy pandas matplotlib scikit-learn tensorflow

#importing Dataset from YAHOO
import yfinance as yf

# Define the stock symbol and time range
ticker = "RVNL.NS"  # Example: Apple stock
start_date = "2023-06-01"
end_date = "2025-02-05"

# Download historical data
data = yf.download(ticker, start=start_date, end=end_date)

# Save data to a CSV file
data.to_csv("RVNL.NS_Historical_Data.csv")

print(data.head())  # Display first few rows

#Downloading some important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns


## RVNL STOCK OVER TIME 

# Plot stock price
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Close Price')
plt.title("RVNL Stock Price Over Time")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

# Plot volume trends
plt.figure(figsize=(12, 4))
plt.plot(df['Volume'], label='Volume', color='orange')
plt.title("RVNL Trading Volume Over Time")
plt.legend()
plt.show()

## checking data sets 

# Create Moving Averages
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()

# Relative Strength Index (RSI)
def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = compute_rsi(data)

# Moving Average Convergence Divergence (MACD)
data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()

print(data.tail())  # Display with new features


## Actual vs Predicted stock
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Prepare features and target
features = ['Open', 'High', 'Low', 'Volume', 'SMA_5', 'SMA_20', 'RSI', 'MACD']
X = data[features].dropna()
y = data['Close'].loc[X.index]  # Matching indices

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scaling data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.05)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
# Calculate RMSE without the 'squared' parameter
# and take the square root to get the RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae}, RMSE: {rmse}")

# Plot Predictions vs Actual
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label="Actual", color='blue')
plt.plot(y_test.index, y_pred, label="Predicted", color='red', linestyle='dashed')
plt.title("Actual vs Predicted Stock Prices")
plt.legend()
plt.show()

pip install newsapi-python nltk vaderSentiment beautifulsoup4 requests

import requests
from newsapi import NewsApiClient

# Get API Key from https://newsapi.org/
NEWS_API_KEY = "7796cfcd780249b49aa5bdcc914e368d"

# Initialize NewsAPI
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# Fetch news related to RVNL
articles = newsapi.get_everything(q="RVNL", language="en", sort_by="publishedAt", page_size=10)

# Extract headlines
news_headlines = [article['title'] for article in articles['articles']]
print(news_headlines)

import pandas as pd
import yfinance as yf

# Load RVNL Stock Data
stock_data = yf.download("RVNL.NS", start="2010-01-01", end="2024-02-01")
stock_data.to_csv("RVNL_Stock.csv")

print(stock_data.head())  # Display first few rows

pip install streamlit

print(df.columns)

import pandas as pd
import numpy as np
import yfinance as yf

# Define the stock symbol and time range
ticker = "RVNL.NS"
start_date = "2024-05-08"
end_date = "2025-02-05"

# Download historical data from Yahoo Finance
stock_data = yf.download(ticker, start=start_date, end=end_date)

# Reset index to move 'Date' from index to column
stock_data.reset_index(inplace=True)

# Save the downloaded data
stock_data.to_csv("RVNL.NS_Historical_Data.csv", index=False)

# Display first few rows
print("✅ Data downloaded & saved successfully!")
print(stock_data.head())

## Actual vs Predicted stock prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import keras_tuner as kt  # Hyperparameter Tuning

# Load stock data
df = pd.read_csv("RVNL.NS_Historical_Data.csv")

# Ensure 'Date' column is in correct datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
df = df.sort_values(by="Date")

# Normalize Close Prices
scaler = MinMaxScaler(feature_range=(0, 1))
df['Close_Scaled'] = scaler.fit_transform(df[['Close']])

# Prepare Data for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 50  # Lookback period
X, y = create_sequences(df['Close_Scaled'].values, seq_length)

# Train-Test Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape for LSTM (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Function to Build LSTM Model for Tuning
def build_lstm_model(hp):
    model = Sequential()
    model.add(LSTM(hp.Int('units_1', min_value=32, max_value=256, step=32),
                   return_sequences=True, input_shape=(seq_length, 1)))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))

    model.add(LSTM(hp.Int('units_2', min_value=32, max_value=256, step=32),
                   return_sequences=False))
    model.add(Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))

    model.add(Dense(1))  # Output Layer

    model.compile(optimizer=hp.Choice('optimizer', ['adam', 'sgd']),
                  loss='mse')
    return model

# Hyperparameter Tuning using Keras Tuner
tuner = kt.RandomSearch(
    build_lstm_model,
    objective='val_loss',
    max_trials=10,  # Number of different models to try
    executions_per_trial=1,
    directory='lstm_tuning',
    project_name='stock_price_prediction'
)

tuner.search(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# Get the Best Model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

# Train the Best Model
history = best_model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Predictions on Test Data
y_pred = best_model.predict(X_test)

# Convert Predictions Back to Original Scale
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate Model Performance
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_rescaled, y_pred_rescaled)

print(f"📌 Model Performance:")
print(f"✅ Mean Squared Error (MSE): {mse:.4f}")
print(f"✅ Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"✅ R² Score: {r2:.4f}")

# Predict Future Prices
future_days = 30
last_seq = df['Close_Scaled'].values[-seq_length:]
predictions = []

for _ in range(future_days):
    last_seq_reshaped = last_seq.reshape((1, seq_length, 1))
    next_price = best_model.predict(last_seq_reshaped)[0, 0]
    predictions.append(next_price)
    last_seq = np.append(last_seq[1:], next_price)

# Convert Predictions Back to Original Scale
predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# Create Future Date Range
future_dates = pd.date_range(start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=future_days, freq="D")

# Create DataFrame for Predictions
df_predicted = pd.DataFrame({
    "Date": future_dates,
    "Open": predicted_prices * 0.98,
    "High": predicted_prices * 1.02,
    "Low": predicted_prices * 0.97,
    "Close": predicted_prices
})

# Candlestick Chart with Predictions
fig = go.Figure()

# Historical Data
fig.add_trace(go.Candlestick(
    x=df["Date"],
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="Historical Data"
))

# Predicted Data
fig.add_trace(go.Candlestick(
    x=df_predicted["Date"],
    open=df_predicted["Open"],
    high=df_predicted["High"],
    low=df_predicted["Low"],
    close=df_predicted["Close"],
    name="Predicted Data",
    increasing_line_color='cyan', decreasing_line_color='gray'
))

# Customize Layout
fig.update_layout(
    title="RVNL.NS Stock Price Candlestick Chart with Optimized LSTM Predictions",
    xaxis_title="Date",
    yaxis_title="Stock Price (INR)",
    xaxis_rangeslider_visible=False
)

# Show Plot
fig.show()

## Residual Distribution

import seaborn as sns
import matplotlib.pyplot as plt

# Calculate Residuals (Errors in Predictions)
residuals = y_test_rescaled.flatten() - y_pred_rescaled.flatten()

# ✅ Plot the Residuals Histogram
plt.figure(figsize=(10, 5))
sns.histplot(residuals, bins=50, kde=True, color="red")
plt.title("Residuals Distribution (RVNL Stock Prediction)")
plt.xlabel("Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.axvline(0, color='black', linestyle='--')  # Reference line at zero
plt.show()

## MACD indicator

import pandas as pd
import plotly.express as px

# Assuming df contains RVNL stock data with 'Date' and 'Close' columns
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# ✅ Plot MACD and Signal Line
fig = px.line(df, x="Date", y=["MACD", "Signal_Line"],
              title="MACD Indicator for RVNL Stock",
              labels={"value": "MACD / Signal Line", "Date": "Date"})

fig.show()


pip install numpy pandas matplotlib plotly tensorflow scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 1: Load data
ticker = "RVNL.NS"
start_date = "2024-05-08"
end_date = "2025-02-05"
data = yf.download(ticker, start=start_date, end=end_date)

# Step 2: Preprocess (Close Price only)
df = data[['Close']].copy()
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Step 3: Create sequences
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)

# Reshape input to [samples, time steps, features] for LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Step 4: Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 5: Train the model
model.fit(X, y, epochs=25, batch_size=32)

# Step 6: Predict the next price
last_60_days = scaled_data[-60:]
X_test = np.reshape(last_60_days, (1, time_step, 1))
predicted_scaled_price = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_scaled_price)

print(f"\n🔮 Predicted Next Closing Price: ₹{predicted_price[0][0]:.2f}")

# Step 7: Plot predictions vs actual (for existing data)
train_predict = model.predict(X)
train_predict_plot = np.empty_like(scaled_data)
train_predict_plot[:, :] = np.nan
train_predict_plot[time_step:len(train_predict)+time_step, 0] = train_predict[:, 0]

# Inverse transform to real values
train_predict_real = scaler.inverse_transform(train_predict)

# Plotting
plt.figure(figsize=(14,6))
plt.plot(df.index, df['Close'], label="Actual Price", color="blue")
plt.plot(df.index[time_step:], train_predict_real, label="LSTM Prediction", color="red")
plt.title("RVNL Stock Price Prediction using LSTM")
plt.xlabel("Date")
plt.ylabel("Stock Price (INR)")
plt.legend()
plt.grid()
plt.show()

# Check what columns actually exist in df
print("Current DataFrame columns:", df.columns.tolist())

# Check for NaNs in each column
print("NaN counts:\n", df[['Close']].isna().sum())  # Close should always exist
print("Checking if MA50, MA200, RSI were created:")

for col in ['MA50', 'MA200', 'RSI']:
    if col not in df.columns:
        print(f"❌ Column '{col}' is missing!")
    else:
        print(f"✅ Column '{col}' exists with {df[col].isna().sum()} NaNs")


## RVNL STOCK prediction using LSTM
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 1: Load data
ticker = "RVNL.NS"
start_date = "2023-06-01"
end_date = "2025-02-05"
df = yf.download(ticker, start=start_date, end=end_date)

# ✅ Fix MultiIndex by flattening it
df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

# Step 2: Add Technical Indicators
df['MA50'] = df['Close'].rolling(window=50).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()

delta = df['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# Drop NaNs only after indicators are added
feature_cols = ['Close', 'MA50', 'MA200', 'RSI']
df.dropna(subset=feature_cols, inplace=True)

# ✅ Check if data is empty
if df.empty:
    raise ValueError("DataFrame is empty after dropping NaNs. Please adjust date range or indicators.")

# Step 3: Prepare features
data = df[feature_cols]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create LSTM sequences
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i - time_step:i])
        y.append(dataset[i, 0])  # Predict 'Close'
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)
X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

# Step 4: LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, y, epochs=20, batch_size=32)

# Step 5: Predictions
predicted = model.predict(X)
predicted_prices = scaler.inverse_transform(
    np.hstack((predicted, np.zeros((len(predicted), len(feature_cols) - 1)))
))[:, 0]

# Step 6: Add Predictions and Signals
df = df.iloc[time_step:]
df['Predicted'] = predicted_prices

def signal_generator(row):
    if row['Close'] < row['MA50'] and row['Predicted'] > row['Close'] and row['RSI'] < 30:
        return 'Buy'
    elif row['Close'] > row['MA50'] and row['Predicted'] < row['Close'] and row['RSI'] > 70:
        return 'Sell'
    else:
        return 'Hold'

df['Signal'] = df.apply(signal_generator, axis=1)

# Step 7: Plot
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Close'], label='Actual Price', alpha=0.6)
plt.plot(df.index, df['Predicted'], label='LSTM Predicted', color='orange')

# Add Buy/Sell markers
buy_signals = df[df['Signal'] == 'Buy']
sell_signals = df[df['Signal'] == 'Sell']
plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal', s=100)
plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal', s=100)

plt.title("RVNL Stock Price with LSTM Predictions and Buy/Sell Signals")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 📦 Install Prophet if needed
# !pip install prophet

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from prophet import Prophet

# Step 1: Load data
ticker = "RVNL.NS"
start_date = "2023-06-01"
end_date = "2025-02-05"
df = yf.download(ticker, start=start_date, end=end_date)

# Step 2: Flatten MultiIndex columns (if they exist)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

# Step 3: Prepare DataFrame for Prophet
df_prophet = df.reset_index()[['Date', 'Close']]
df_prophet.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

# Ensure the 'y' column is numeric and clean
df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
df_prophet.dropna(inplace=True)

# Step 4: Build and train Prophet model
model = Prophet(daily_seasonality=True)
model.fit(df_prophet)

# Step 5: Create future DataFrame (e.g. 30 days into the future)
future = model.make_future_dataframe(periods=30)

# Step 6: Make forecast
forecast = model.predict(future)

# Step 7: Plot forecast
fig = model.plot(forecast)
plt.title("📈 RVNL Stock Price Forecast using Prophet")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.grid(True)
plt.show()

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# Step 1: Download Data
df = yf.download('RVNL.NS', start='2023-06-01', end='2024-06-01')['Close']
df.dropna(inplace=True)

# Step 2: Grid Search for Best ARIMA Order (p, d, q) using AIC
best_aic = float("inf")
best_order = None
best_model = None

# Define ranges
p_range = range(0, 4)
d_range = range(0, 3)
q_range = range(0, 4)

print("Tuning ARIMA hyperparameters...")

for p in p_range:
    for d in d_range:
        for q in q_range:
            try:
                model = ARIMA(df, order=(p, d, q))
                model_fit = model.fit()
                current_aic = model_fit.aic
                if current_aic < best_aic:
                    best_aic = current_aic
                    best_order = (p, d, q)
                    best_model = model_fit
            except:
                continue

print(f"✅ Best ARIMA order: {best_order} with AIC: {best_aic:.2f}")

# Step 3: Forecast future values
forecast_steps = 30  # next 30 business days
forecast = best_model.forecast(steps=forecast_steps)

# Step 4: Plot
plt.figure(figsize=(12, 5))
plt.plot(df.index, df, label='Historical')
future_dates = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='B')[1:]
plt.plot(future_dates, forecast, label='Forecast', color='red')
plt.title(f'ARIMA Forecast (order={best_order}) for RVNL')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# Step 1: Load data
df = yf.download('RVNL.NS', start='2023-06-01', end='2024-06-01')['Close']
df.dropna(inplace=True)

# Step 2: ARIMA grid search for AIC and BIC
p_range = range(0, 4)
d_range = range(0, 3)
q_range = range(0, 4)

results = []
print("🔍 Tuning ARIMA hyperparameters using AIC and BIC...")

for p in p_range:
    for d in d_range:
        for q in q_range:
            try:
                model = ARIMA(df, order=(p, d, q))
                model_fit = model.fit()
                aic = model_fit.aic
                bic = model_fit.bic
                results.append({'order': (p, d, q), 'aic': aic, 'bic': bic})
                print(f"✔️ Tried {(p,d,q)} -> AIC: {aic:.2f}, BIC: {bic:.2f}")
            except:
                continue

# Step 3: Sort and pick best model
results_df = pd.DataFrame(results)
best_row = results_df.loc[results_df['aic'].idxmin()]
print(f"\n✅ Best ARIMA order based on AIC: {best_row['order']} (AIC: {best_row['aic']:.2f}, BIC: {best_row['bic']:.2f})")

# Step 4: Plot AIC and BIC
plt.figure(figsize=(14, 6))

# AIC
plt.subplot(1, 2, 1)
plt.plot(results_df['order'].astype(str), results_df['aic'], marker='o', label='AIC')
plt.xticks(rotation=90)
plt.title('AIC Scores by ARIMA Order')
plt.xlabel('ARIMA Order (p,d,q)')
plt.ylabel('AIC')
plt.grid(True)

# BIC
plt.subplot(1, 2, 2)
plt.plot(results_df['order'].astype(str), results_df['bic'], marker='s', color='orange', label='BIC')
plt.xticks(rotation=90)
plt.title('BIC Scores by ARIMA Order')
plt.xlabel('ARIMA Order (p,d,q)')
plt.ylabel('BIC')
plt.grid(True)

plt.tight_layout()
plt.show()

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Download stock data
df = yf.download('RVNL.NS', start='2023-06-01', end='2024-06-01')['Close']
df.dropna(inplace=True)

# Step 2: Calculate rolling mean (choose window size)
rolling_window = 30  # You can change this to 7, 14, 50, etc.
df_rolling = df.rolling(window=rolling_window).mean()

# Step 3: Plot actual vs rolling mean
plt.figure(figsize=(14, 6))
plt.plot(df.index, df, label='Actual Price', color='blue', linewidth=2)
plt.plot(df_rolling.index, df_rolling, label=f'{rolling_window}-Day Rolling Mean', color='orange', linewidth=2)
plt.title(f'RVNL.NS: Actual Price vs {rolling_window}-Day Rolling Mean')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


