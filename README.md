# 🚆 RVNL Stock Price Prediction using Machine Learning
This repository presents a **research-driven machine learning project** focused on forecasting the closing prices of RVNL (Rail Vikas Nigam Limited) stocks. By integrating both **deep learning (LSTM)** and **XGBoost regression**, the model aims to simulate real-world trading scenarios and enhance predictive performance using financial technical indicators.

---

## 📈 Project Objective

To build accurate and interpretable models that forecast stock closing prices for RVNL using a hybrid approach, incorporating technical indicators to improve signal quality and decision-making in stock trading.

---

## 🧠 Models Used

- 🔹 **Long Short-Term Memory (LSTM)**: For capturing temporal dependencies in stock time-series data.
- 🔹 **XGBoost Regression**: A powerful ensemble method used for tabular feature-based predictions.

---

## 📊 Technical Indicators Used

- **RSI (Relative Strength Index)**  
- **MACD (Moving Average Convergence Divergence)**  
- **SMA (Simple Moving Average)**  

These indicators serve as engineered features to improve model performance.

---

## 📦 Tech Stack

| Tool | Purpose |
|------|---------|
| `Python` | Core programming language |
| `Pandas`, `NumPy` | Data manipulation |
| `Matplotlib`, `Seaborn` | Visualization |
| `Sklearn`, `XGBoost` | Machine learning models |
| `TensorFlow / Keras` | Deep learning (LSTM model) |

---

## 📂 Project Structure

RVNL-PREDICTION/
│
├── data/                     # Raw and processed datasets
│   ├── rvnl_stock.csv
│   └── technical_indicators.csv
│
├── notebooks/                # Jupyter notebooks for analysis and modeling
│   ├── LSTM_Model.ipynb
│   └── XGBoost_Model.ipynb
│
├── models/                   # Saved models (if any)
│   └── lstm_model.h5
│
├── visuals/                  # Graphs and plots used in the report
│   ├── closing_price_plot.png
│   └── macd_rsi_visuals.png
│
├── README.md                 # Project overview and documentation
├── requirements.txt          # Python dependencies
├── utils.py                  # Helper functions for preprocessing, evaluation
└── main.py                   # Entry point to run the training and prediction pipeline

### 🔍 Model Performance

![RVNL STOCK PRICE WITH LSTM](visuals/RVNL_STOCK_PRICE_WITH_LSTM.jpeg)

![Daily Resturn of RVNL stock](visuals/Daily_Return_Distribution.jpeg)

---

## 📌 Highlights

- ✅ Clean and modular code
- 📈 Visualizations of actual vs predicted prices
- ⚙️ Backtested with indicators to mimic real trading scenarios
- 📉 Evaluation metrics: RMSE, MAE, and R² score

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Satyabrat2005/RVNL-PREDICTION.git
   cd RVNL-PREDICTION

2. Install dependencies:
    pip install -r requirements.txt

Run Jupyter notebooks or scripts inside **/notebooks** to explore the models.


### 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.




