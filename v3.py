# app.py
# Professional Indian Stock Forecast Dashboard
# Streamlit + Yahoo Finance + Auto ARIMA + Plotly
# Forecasts through June 2027

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px

from io import BytesIO
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Indian Stock Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.metric-card {
    background-color: #0E1117;
    border: 1px solid #262730;
    border-radius: 12px;
    padding: 15px;
}

h1 {
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.title("📈 Professional Indian Stock Forecast Dashboard")

st.markdown(
"""
Forecast Indian stocks using:

✅ Yahoo Finance Data  
✅ Auto ARIMA Model  
✅ RMSE Validation  
✅ Confidence Intervals  
✅ Interactive Plotly Charts  
✅ Forecast Until June 2027
"""
)

# ---------------------------------------------------
# STOCK LIST
# ---------------------------------------------------

stock_dict = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "Larsen & Toubro": "LT.NS",
    "Asian Paints": "ASIANPAINT.NS"
}

colA, colB = st.columns([2, 1])

with colA:
    company = st.selectbox(
        "Select NSE Stock",
        list(stock_dict.keys())
    )

with colB:
    run_forecast = st.button(
        "Generate Forecast",
        use_container_width=True
    )

ticker = stock_dict[company]

# ---------------------------------------------------
# MAIN LOGIC
# ---------------------------------------------------

if run_forecast:

    with st.spinner("Downloading 5 Years of Data..."):

        df = yf.download(
            ticker,
            period="5y",
            auto_adjust=True,
            progress=False
        )

    if df.empty:
        st.error("No data found.")
        st.stop()

    prices = df["Close"].dropna()

    # ----------------------------------------------
    # DATA OVERVIEW
    # ----------------------------------------------

    st.subheader("Historical Data")

    st.dataframe(
        prices.tail(15),
        use_container_width=True
    )

    # ----------------------------------------------
    # TRAIN TEST SPLIT
    # ----------------------------------------------

    train_size = int(len(prices) * 0.80)

    train = prices[:train_size]
    test = prices[train_size:]

    st.info("Running Auto ARIMA Model...")

    model = auto_arima(
        train,
        seasonal=False,
        suppress_warnings=True,
        stepwise=True,
        error_action="ignore"
    )

    predictions = model.predict(
        n_periods=len(test)
    )

    rmse = np.sqrt(
        mean_squared_error(
            test,
            predictions
        )
    )

    # ----------------------------------------------
    # FINAL MODEL
    # ----------------------------------------------

    final_model = auto_arima(
        prices,
        seasonal=False,
        suppress_warnings=True,
        stepwise=True
    )

    last_date = prices.index[-1]

    target_date = pd.Timestamp(
        "2027-06-30"
    )

    forecast_days = (
        target_date - last_date
    ).days

    if forecast_days <= 0:
        st.error(
            "Target date has already passed."
        )
        st.stop()

    forecast, conf_int = final_model.predict(
        n_periods=forecast_days,
        return_conf_int=True
    )

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="B"
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": forecast[:len(future_dates)],
        "Lower_95": conf_int[:len(future_dates),0],
        "Upper_95": conf_int[:len(future_dates),1]
    })

    # ----------------------------------------------
    # METRICS
    # ----------------------------------------------

    current_price = float(prices.iloc[-1])

    future_price = float(
        forecast_df.iloc[-1]["Forecast"]
    )

    expected_return = (
        (future_price - current_price)
        / current_price
    ) * 100

    returns = prices.pct_change().dropna()

    annual_volatility = (
        returns.std() * np.sqrt(252)
    ) * 100

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Current Price",
        f"₹{current_price:.2f}"
    )

    col2.metric(
        "Forecast Price",
        f"₹{future_price:.2f}"
    )

    col3.metric(
        "Expected Return",
        f"{expected_return:.2f}%"
    )

    col4.metric(
        "RMSE",
        f"{rmse:.2f}"
    )

    col5.metric(
        "Volatility",
        f"{annual_volatility:.2f}%"
    )

    # ----------------------------------------------
    # SIGNAL
    # ----------------------------------------------

    if expected_return > 20:
        st.success(
            "📈 Bullish Long-Term Forecast"
        )

    elif expected_return > 0:
        st.info(
            "📊 Moderately Positive Forecast"
        )

    else:
        st.warning(
            "📉 Bearish Forecast"
        )

    # ----------------------------------------------
    # HISTORICAL CHART
    # ----------------------------------------------

    st.subheader("Historical Price Trend")

    fig_hist = px.line(
        x=prices.index,
        y=prices.values,
        labels={
            "x":"Date",
            "y":"Price"
        }
    )

    fig_hist.update_layout(
        height=500,
        template="plotly_dark"
    )

    st.plotly_chart(
        fig_hist,
        use_container_width=True
    )

    # ----------------------------------------------
    # FORECAST CHART
    # ----------------------------------------------

    st.subheader(
        "Forecast Through June 2027"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices.values,
            mode="lines",
            name="Historical"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Forecast"],
            mode="lines",
            name="Forecast"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Upper_95"],
            line=dict(width=0),
            showlegend=False
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Lower_95"],
            fill="tonexty",
            line=dict(width=0),
            name="95% Confidence"
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=650,
        title=f"{ticker} Forecast"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ----------------------------------------------
    # ACTUAL VS PREDICTED
    # ----------------------------------------------

    st.subheader(
        "Model Validation"
    )

    validation = go.Figure()

    validation.add_trace(
        go.Scatter(
            x=test.index,
            y=test.values,
            name="Actual"
        )
    )

    validation.add_trace(
        go.Scatter(
            x=test.index,
            y=predictions,
            name="Predicted"
        )
    )

    validation.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(
        validation,
        use_container_width=True
    )

    # ----------------------------------------------
    # ACF PACF
    # ----------------------------------------------

    st.subheader(
        "ACF & PACF Diagnostics"
    )

    colA, colB = st.columns(2)

    with colA:

        fig_acf, ax = plt.subplots(
            figsize=(6,4)
        )

        plot_acf(
            prices,
            ax=ax,
            lags=30
        )

        st.pyplot(fig_acf)

    with colB:

        fig_pacf, ax = plt.subplots(
            figsize=(6,4)
        )

        plot_pacf(
            prices,
            ax=ax,
            lags=30
        )

        st.pyplot(fig_pacf)

    # ----------------------------------------------
    # MONTHLY FORECAST
    # ----------------------------------------------

    st.subheader(
        "Monthly Forecast Summary"
    )

    monthly = (
        forecast_df
        .set_index("Date")
        .resample("ME")
        .mean()
        .reset_index()
    )

    st.dataframe(
        monthly,
        use_container_width=True
    )

    # ----------------------------------------------
    # JUNE 2027
    # ----------------------------------------------

    st.subheader(
        "June 2027 Forecast"
    )

    june2027 = forecast_df[
        forecast_df["Date"].dt.strftime(
            "%Y-%m"
        ) == "2027-06"
    ]

    st.dataframe(
        june2027,
        use_container_width=True
    )

    # ----------------------------------------------
    # DOWNLOADS
    # ----------------------------------------------

    st.subheader(
        "Download Forecast"
    )

    csv = forecast_df.to_csv(
        index=False
    )

    st.download_button(
        "📥 Download CSV",
        csv,
        file_name=f"{ticker}_forecast.csv",
        mime="text/csv"
    )

    excel_buffer = BytesIO()

    with pd.ExcelWriter(
        excel_buffer,
        engine="xlsxwriter"
    ) as writer:

        forecast_df.to_excel(
            writer,
            index=False,
            sheet_name="Forecast"
        )

    st.download_button(
        "📥 Download Excel",
        excel_buffer.getvalue(),
        file_name=f"{ticker}_forecast.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.success(
        f"Forecast completed using ARIMA {final_model.order}"
    )
