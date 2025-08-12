# 📈 Event-Driven ML Stock Prediction

A machine learning mini-project analyzing how public events—specifically earnings announcements—impact short-term stock price direction. This project demonstrates event study methodology, feature engineering from financial time series, and the application of classification models to real-world market behavior.

## 🧠 Objective

To build a supervised ML model that predicts whether a stock's price will rise or fall immediately after an earnings announcement, using engineered features such as volatility, momentum, and volume.

## 🛠️ Tools & Libraries

- Python 3.9+
- `pandas`, `numpy` – data manipulation
- `yfinance` – historical market data
- `scikit-learn` – ML models and evaluation
- `matplotlib`, `seaborn` – visualization
- `shap` – model explainability (optional)
- `streamlit` – interactive dashboard (optional)

## 📊 Features Engineered

- 5-day return before event
- Daily volatility (rolling std)
- RSI (Relative Strength Index)
- 5- and 10-day moving averages
- Volume change vs. 20-day average
- Event day dummy variable

## 🧪 Model

- **Target**: Binary label (1 = stock rises after event, 0 = falls)
- **Model Used**: Logistic Regression and Random Forest
- **Train/Test Split**: TimeSeriesSplit (to avoid look-ahead bias)
- **Metrics**: Accuracy, Precision, Confusion Matrix, ROC Curve

## 📉 Backtest Strategy

A simple trading simulation based on model predictions:
- Long position entered on positive signal
- Held for 3 days post-event
- Compared cumulative return to baseline buy-and-hold

## 💡 Key Results

- Achieved ~68% accuracy on test data (Random Forest)
- Cumulative strategy return outperformed benchmark by ~12% over test window
- SHAP values identified volume and pre-event volatility as key drivers

## 📌 Example Output

![confusion matrix](images/confusion_matrix.png)  
![strategy performance](images/strategy_plot.png)

## 📁 Folder Structure

