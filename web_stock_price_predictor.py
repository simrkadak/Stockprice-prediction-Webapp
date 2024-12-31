import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.title("Stock Price Predictor App")

stock = st.text_input("Enter the Stock ID", "GOOG")

end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

# Fetch stock data
stock_data = yf.download(stock, start, end)

# Load pre-trained model
model = load_model("Latest_stock_price_model.keras")

st.subheader("Stock Data")
st.write(stock_data)

# Calculate moving averages
stock_data['MA_100'] = stock_data['Close'].rolling(100).mean()
stock_data['MA_200'] = stock_data['Close'].rolling(200).mean()
stock_data['MA_250'] = stock_data['Close'].rolling(250).mean()

# plotting function to handle multiple moving averages
def plot_graph(figsize, data, ma_columns, labels):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data['Close'], label="Close Price")
    for ma_column, label in zip(ma_columns, labels):
        ax.plot(data[ma_column], label=label)
    ax.legend()
    return fig

st.subheader('Original Close Price and MA for 250 days')
st.pyplot(plot_graph((15, 6), stock_data, ['MA_250'], ['MA 250 days']))

st.subheader('Original Close Price and MA for 200 days')
st.pyplot(plot_graph((15, 6), stock_data, ['MA_200'], ['MA 200 days']))

st.subheader('Original Close Price and MA for 100 days')
st.pyplot(plot_graph((15, 6), stock_data, ['MA_100'], ['MA 100 days']))

st.subheader('Original Close Price, MA for 100 days, and MA for 250 days')
st.pyplot(plot_graph((15, 6), stock_data, ['MA_100', 'MA_250'], ['MA 100 days', 'MA 250 days']))

# Data preprocessing for prediction
split_len = int(len(stock_data) * 0.7)
test_data = stock_data['Close'][split_len:].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(test_data)

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Make predictions
predictions = model.predict(x_data)
inv_predictions = scaler.inverse_transform(predictions)
inv_y_data = scaler.inverse_transform(y_data)

# Prepare data for plotting
plot_data = pd.DataFrame({
    'Original': inv_y_data.flatten(),
    'Predicted': inv_predictions.flatten()
}, index=stock_data.index[split_len+100:])

st.subheader("Original values vs Predicted values")
st.write(plot_data)

st.subheader('Original Close Price vs Predicted Close Price')
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(stock_data['Close'][:split_len+100], label="Data not used")
ax.plot(plot_data['Original'], label="Original Test Data")
ax.plot(plot_data['Predicted'], label="Predicted Test Data")
ax.legend()
st.pyplot(fig)
