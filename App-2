import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from pmdarima import auto_arima

from io import BytesIO

st.set_page_config(
    page_title="Professional ARIMA Stock Forecast",
    layout="wide"
)

st.title("📈 Indian Stock Forecasting using Auto ARIMA")

st.markdown("""
### Features
- 5 Years Historical Data
- Automatic ARIMA Selection
- ACF & PACF Analysis
- RMSE Accuracy
- Confidence Intervals
- Forecast until June 2027
- CSV & Excel Download
""")

ticker = st.text_input(
    "Enter NSE Ticker",
    value="RELIANCE.NS"
)

if st.button("Generate Forecast"):

    try:

        with st.spinner("Downloading data..."):

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

        st.subheader("Recent Historical Data")
        st.dataframe(prices.tail(20))

        train_size = int(len(prices) * 0.80)

        train = prices[:train_size]
        test = prices[train_size:]

        st.info("Running Auto ARIMA...")

        model = auto_arima(
            train,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore"
        )

        st.success(
            f"Selected ARIMA Order: {model.order}"
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

        st.metric(
            "RMSE",
            round(rmse, 2)
        )

        st.subheader("Actual vs Predicted")

        fig1, ax1 = plt.subplots(
            figsize=(10,5)
        )

        ax1.plot(
            test.index,
            test,
            label="Actual"
        )

        ax1.plot(
            test.index,
            predictions,
            label="Predicted"
        )

        ax1.legend()

        st.pyplot(fig1)

        st.subheader("ACF Plot")

        fig2, ax2 = plt.subplots(
            figsize=(10,4)
        )

        plot_acf(
            prices,
            ax=ax2,
            lags=30
        )

        st.pyplot(fig2)

        st.subheader("PACF Plot")

        fig3, ax3 = plt.subplots(
            figsize=(10,4)
        )

        plot_pacf(
            prices,
            ax=ax3,
            lags=30
        )

        st.pyplot(fig3)

        st.info(
            "Training final model on full dataset..."
        )

        final_model = auto_arima(
            prices,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True
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
                "Target date already reached."
            )
            st.stop()

        forecast, conf_int = final_model.predict(
            n_periods=forecast_days,
            return_conf_int=True
        )

        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_days,
            freq="D"
        )

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast": forecast,
            "Lower_95": conf_int[:,0],
            "Upper_95": conf_int[:,1]
        })

        june2027 = forecast_df[
            forecast_df["Date"].dt.strftime("%Y-%m")
            == "2027-06"
        ]

        st.subheader(
            "June 2027 Forecast Values"
        )

        st.dataframe(june2027)

        st.subheader(
            "Historical + Forecast"
        )

        fig4, ax4 = plt.subplots(
            figsize=(12,6)
        )

        ax4.plot(
            prices.index,
            prices,
            label="Historical"
        )

        ax4.plot(
            forecast_df["Date"],
            forecast_df["Forecast"],
            label="Forecast"
        )

        ax4.fill_between(
            forecast_df["Date"],
            forecast_df["Lower_95"],
            forecast_df["Upper_95"],
            alpha=0.3
        )

        ax4.legend()

        st.pyplot(fig4)

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

    except Exception as e:
        st.error(str(e))
