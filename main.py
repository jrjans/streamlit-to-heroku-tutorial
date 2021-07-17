# Build a Stock Prediction Web App In Python
# https://www.youtube.com/watch?v=0E_31WqVzCY

# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Title of the app
st.title('Stock Forecast App')

# Stocks that can be selected + Selection box that will have the stocks
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

# Slider to select number of years that will be predicted
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Load the stock data
@st.cache
def load_data(ticker):
	data = yf.download(ticker, START, TODAY)
	data.reset_index(inplace=True)
	return data

data_load_state = st.text("Load Data...")
data = load_data(selected_stock)
data_load_state.text("Loading Data...done!")

# Plot the data
st.subheader('Raw Data')
st.write(data.tail())

# Plot the data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
	fig.layout.update(title_text="Time Series Data", xaxis_rangeslides_visible=True)
	st.plotly_chart(fig)

plot_raw_data()

#----------------------------------------------------
# Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Create a FB Prophet Model
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(data.tail())

