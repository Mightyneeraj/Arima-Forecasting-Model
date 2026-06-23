import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- Page Configuration ---
st.set_page_config(page_title="Advanced Forecast Modeler", page_icon="📈", layout="wide")

# --- Header ---
st.title("📈 Advanced Forecast Modeler")
st.markdown("""
This tool analyzes your historical data and projects future values. 
**Requirements:** Your Excel file must contain at least one column with **Dates** and one column with **Numerical Values**.
""")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload Excel File (.xlsx or .xls)", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Load Data
        df = pd.read_excel(uploaded_file)
        
        st.sidebar.header("2. Configure Model")
        
        # Identify column types
        date_cols = df.select_dtypes(include=['datetime64', 'object']).columns.tolist()
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if not num_cols:
            st.error("No numerical columns found. A target variable is required for forecasting.")
            st.stop()
            
        date_col = st.sidebar.selectbox("Select Date Column:", date_cols)
        target_col = st.sidebar.selectbox("Select Target Variable to Forecast:", num_cols)
        agg_method = st.sidebar.selectbox("Aggregation (if multiple entries per month):", ["Sum", "Mean"])
        
        if st.sidebar.button("Generate Forecast"):
            
            # --- Data Preprocessing ---
            # Attempt to convert selected date column to datetime
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col, target_col])
            
            if df.empty:
                st.error("Data is empty after dropping invalid dates/values.")
                st.stop()
                
            # Sort, set index, and resample to monthly data
            df = df.sort_values(by=date_col)
            df.set_index(date_col, inplace=True)
            
            if agg_method == "Sum":
                ts_data = df[target_col].resample('MS').sum() # Monthly Start
            else:
                ts_data = df[target_col].resample('MS').mean()
                
            # --- UI Layout: Tabs ---
            tab1, tab2, tab3 = st.tabs(["📊 Future Forecast (June 2027)", "📉 Historical Trends", "📋 Data Distribution"])
            
            # --- TAB 1: Forecasting ---
            with tab1:
                st.subheader(f"Forecast Projection for {target_col}")
                
                # Fit the Exponential Smoothing Model
                try:
                    # Using an additive trend model; you can adjust seasonality if needed
                    model = ExponentialSmoothing(ts_data, trend='add', seasonal=None, initialization_method="estimated")
                    fit_model = model.fit()
                    
                    # Calculate steps to June 2027
                    target_date = pd.to_datetime("2027-06-01")
                    last_date = ts_data.index[-1]
                    
                    if last_date >= target_date:
                        st.warning("Your historical data already reaches or exceeds June 2027.")
                        steps = 12 # Default to 12 months forward
                    else:
                        steps = (target_date.year - last_date.year) * 12 + target_date.month - last_date.month
                    
                    # Generate Forecast
                    forecast = fit_model.forecast(steps)
                    forecast.index = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=steps, freq='MS')
                    
                    # Plotly Interactive Chart
                    fig = go.Figure()
                    
                    # Historical Line
                    fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values, mode='lines+markers', name='Historical Data', line=dict(color='#1f77b4')))
                    
                    # Forecast Line
                    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode='lines+markers', name='Forecast', line=dict(color='#ff7f0e', dash='dot')))
                    
                    # Highlight June 2027
                    if target_date in forecast.index:
                        jun_27_val = forecast.loc[target_date]
                        fig.add_annotation(x=target_date, y=jun_27_val, text=f"Jun 2027: {jun_27_val:,.2f}", showarrow=True, arrowhead=1, ax=-40, ay=-40)
                    
                    fig.update_layout(title="Historical vs. Forecasted Values", xaxis_title="Date", yaxis_title=target_col, hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Forecast Stats
                    col1, col2 = st.columns(2)
                    col1.metric("Last Recorded Value", f"{ts_data.values[-1]:,.2f}")
                    if target_date in forecast.index:
                        col2.metric("Predicted Value (Jun 2027)", f"{forecast.loc[target_date]:,.2f}")
                    
                except Exception as e:
                    st.error(f"Could not generate forecast. Error: {e}")

            # --- TAB 2: Historical Trends ---
            with tab2:
                st.subheader("Historical Timeline")
                fig_hist = px.line(ts_data, x=ts_data.index, y=ts_data.values, labels={'y': target_col, 'index': 'Date'}, markers=True)
                st.plotly_chart(fig_hist, use_container_width=True)
                
                st.write("Monthly Aggregated Data")
                st.dataframe(ts_data.reset_index().style.format({target_col: "{:.2f}"}), use_container_width=True)

            # --- TAB 3: Data Distribution ---
            with tab3:
                st.subheader("Distribution & Variance Analysis")
                col1, col2 = st.columns([2, 1])
                
                raw_target_data = df[target_col].dropna()
                mean_val = raw_target_data.mean()
                median_val = raw_target_data.median()
                diff_percent = 0 if mean_val == 0 else abs(mean_val - median_val) / abs(mean_val) * 100
                
                with col1:
                    fig_dist = px.histogram(df, x=target_col, nbins=30, marginal="box", title=f"Distribution of {target_col}")
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                with col2:
                    st.metric("Mean", f"{mean_val:,.2f}")
                    st.metric("Median", f"{median_val:,.2f}")
                    st.metric("Mean/Median Diff (%)", f"{diff_percent:.1f}%")
                    
                    if diff_percent <= 10:
                        st.success("Normal Distribution (Low Skew)")
                    else:
                        st.warning("Skewed Distribution detected. Averages may be distorted by outliers.")

    except Exception as e:
        st.error(f"Error processing the file: {e}")
