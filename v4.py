
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from io import BytesIO
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

st.set_page_config(page_title="Indian Stock Forecast Dashboard", page_icon="📈", layout="wide")

st.title("📈 Indian Stock Forecast Dashboard")
st.caption("5-Year Historical Data • ARIMA Forecast • Confidence Intervals • June 2027 Projection")

stocks = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "SBI": "SBIN.NS",
    "ITC": "ITC.NS",
    "Bharti Airtel": "BHARTIARTL.NS"
}

company = st.selectbox("Select Stock", list(stocks.keys()))
ticker = stocks[company]

if st.button("Generate Forecast", use_container_width=True):

    with st.spinner("Downloading data..."):
        df = yf.download(ticker, period="5y", auto_adjust=True, progress=False)

    if df.empty:
        st.error("No data received from Yahoo Finance.")
        st.stop()

    close_data = df["Close"]
    if isinstance(close_data, pd.DataFrame):
        prices = close_data.iloc[:, 0]
    else:
        prices = close_data

    prices = pd.Series(prices).dropna()

    if len(prices) < 100:
        st.error("Insufficient data.")
        st.stop()

    train_size = int(len(prices) * 0.8)
    train = prices[:train_size]
    test = prices[train_size:]

    model = ARIMA(train, order=(5,1,0))
    fitted = model.fit()

    pred = fitted.forecast(steps=len(test))

    rmse = np.sqrt(mean_squared_error(test, pred))

    final_model = ARIMA(prices, order=(5,1,0)).fit()

    last_date = prices.index[-1]
    target_date = pd.Timestamp("2027-06-30")

    business_days = len(pd.date_range(start=last_date, end=target_date, freq="B"))

    forecast_res = final_model.get_forecast(steps=business_days)

    forecast = forecast_res.predicted_mean
    conf = forecast_res.conf_int()

    future_dates = pd.date_range(
        start=last_date + pd.offsets.BDay(1),
        periods=business_days,
        freq="B"
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": np.asarray(forecast),
        "Lower_95": np.asarray(conf.iloc[:,0]),
        "Upper_95": np.asarray(conf.iloc[:,1])
    })

    current_price = float(prices.iloc[-1])
    future_price = float(forecast_df["Forecast"].iloc[-1])

    expected_return = ((future_price-current_price)/current_price)*100

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Current Price", f"₹{current_price:,.2f}")
    c2.metric("Forecast Price", f"₹{future_price:,.2f}")
    c3.metric("Expected Return", f"{expected_return:.2f}%")
    c4.metric("RMSE", f"{rmse:.2f}")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=prices.index,
        y=prices.values,
        mode="lines",
        name="Historical"
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Forecast"],
        mode="lines",
        name="Forecast"
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Upper_95"],
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Lower_95"],
        fill="tonexty",
        line=dict(width=0),
        name="95% Confidence Interval"
    ))

    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model Validation")

    val_fig = go.Figure()
    val_fig.add_trace(go.Scatter(x=test.index, y=test.values, name="Actual"))
    val_fig.add_trace(go.Scatter(x=test.index, y=pred, name="Predicted"))
    st.plotly_chart(val_fig, use_container_width=True)

    st.subheader("ACF / PACF")

    a,b = st.columns(2)

    with a:
        fig_acf, ax = plt.subplots()
        plot_acf(prices, lags=30, ax=ax)
        st.pyplot(fig_acf)

    with b:
        fig_pacf, ax = plt.subplots()
        plot_pacf(prices, lags=30, ax=ax)
        st.pyplot(fig_pacf)

    st.subheader("June 2027 Forecast")

    june = forecast_df[forecast_df["Date"].dt.strftime("%Y-%m") == "2027-06"]
    st.dataframe(june, use_container_width=True)

    csv_data = forecast_df.to_csv(index=False)

    st.download_button(
        "Download CSV",
        csv_data,
        file_name=f"{ticker}_forecast.csv"
    )

    excel_buffer = BytesIO()

    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        forecast_df.to_excel(writer, index=False)

    st.download_button(
        "Download Excel",
        excel_buffer.getvalue(),
        file_name=f"{ticker}_forecast.xlsx"
    )
