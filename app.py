import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from prophet import Prophet
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="📊 Unemployment Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
df = pd.read_csv("unemployment_data.csv")
df.columns = df.columns.str.strip()
df.rename(columns={'Estimated Unemployment Rate (%)': 'Unemployment Rate'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Region'] = df['Region'].astype(str)

# Sidebar Navigation
st.sidebar.title("🔎 Navigation")
nav = st.sidebar.radio("Go to", ["Overview", "Regional Analysis", "Forecasting"])

st.sidebar.markdown("---")


# --- OVERVIEW SECTION ---
if nav == "Overview":
    st.title("📈 Unemployment Analysis Dashboard")
    st.markdown("Analyze and visualize unemployment trends in India with insights on COVID-19 impact and seasonal behavior.")

    st.subheader("🗕️ Unemployment Trend Over Time")
    fig1 = px.line(df, x='Date', y='Unemployment Rate', title='Unemployment Rate Over Time')
    st.plotly_chart(fig1, use_container_width=True)

    # COVID Comparison
    st.subheader("🦠 Impact of COVID-19 on Unemployment")
    pre_covid = df[df['Date'] < '2020-03-01']
    post_covid = df[df['Date'] >= '2020-03-01']
    pre_avg = pre_covid['Unemployment Rate'].mean()
    post_avg = post_covid['Unemployment Rate'].mean()

    col1, col2 = st.columns(2)
    col1.metric("Pre-COVID Avg Unemployment", f"{pre_avg:.2f}%")
    col2.metric("Post-COVID Avg Unemployment", f"{post_avg:.2f}%", delta=f"{post_avg - pre_avg:.2f}")

    st.markdown("---")
    st.subheader("📆 Monthly Average Unemployment Trend")
    monthly_avg = df.groupby('Month')['Unemployment Rate'].mean().reset_index()
    fig_month = px.line(monthly_avg, x='Month', y='Unemployment Rate', markers=True)
    st.plotly_chart(fig_month, use_container_width=True)

    st.markdown("---")
    st.subheader("🗜️ Choropleth Map of Unemployment Over Time")
    df['Date_str'] = df['Date'].dt.strftime('%Y-%m')
    fig_map = px.choropleth(
        df,
        locations='Region',
        locationmode='country names',
        color='Unemployment Rate',
        hover_name='Region',
        animation_frame='Date_str',
        color_continuous_scale='YlOrRd',
        title='Unemployment Rate by Region Over Time',
        scope='asia'
    )
    fig_map.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Bar Graph of Unemployment by Region")
    available_dates = sorted(df['Date'].dt.date.unique())
    selected_date = st.selectbox("Select Date for Bar Chart", available_dates, index=len(available_dates)-1)
    df_bar = df[df['Date'].dt.date == selected_date]
    fig_bar = px.bar(
        df_bar,
        x='Region',
        y='Unemployment Rate',
        color='Region',
        title=f'Unemployment Rate by Region on {selected_date}',
        labels={'Unemployment Rate': 'Rate (%)'},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_bar.update_layout(xaxis_tickangle=45, height=500)
    st.plotly_chart(fig_bar, use_container_width=True)

# --- REGIONAL ANALYSIS SECTION ---
elif nav == "Regional Analysis":
    st.title("📍 Regional Unemployment Analysis")

    region = st.selectbox("Select Region", sorted(df['Region'].unique()))
    filtered = df[df['Region'] == region]

    st.subheader(f"📊 Trend in {region}")
    fig = px.line(filtered, x='Date', y='Unemployment Rate', color='Region', title=f'Unemployment Rate in {region}')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🗓️ Boxplot of Unemployment Rates")
    fig_box = px.box(filtered, x='Region', y='Unemployment Rate')
    st.plotly_chart(fig_box, use_container_width=True)

# --- FORECASTING SECTION ---
elif nav == "Forecasting":
    st.title("📉 Forecasting Future Unemployment Rate")
    st.markdown("Select a forecasting model to predict unemployment for the next 12 months.")

    model_type = st.selectbox("Select Forecasting Model", ["Random Forest", "Prophet"])

    if model_type == "Random Forest":
        try:
            model_rf = joblib.load("rf_model.pkl")
            df['Time'] = range(len(df))
            future_time = range(len(df), len(df)+12)
            future_months = [(m % 12) + 1 for m in future_time]
            X_future = pd.DataFrame({'Month': future_months, 'Time': future_time})
            preds = model_rf.predict(X_future)
            future_dates = pd.date_range(start=df['Date'].max(), periods=12, freq='M')
            forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': preds})

            st.subheader("📈 Forecast using Random Forest")
            fig_rf = px.line(forecast_df, x='Date', y='Forecast', title="Next 12-Month Forecast")
            st.plotly_chart(fig_rf, use_container_width=True)

        except FileNotFoundError:
            st.error("❌ Random Forest model not found. Please train and save it as `models/rf_model.pkl`.")

    else:
        st.subheader("📈 Forecast using Prophet")
        df_prophet = df.rename(columns={'Date': 'ds', 'Unemployment Rate': 'y'})
        model = Prophet()
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)
        fig_prophet = model.plot(forecast)
        st.pyplot(fig_prophet)

        st.markdown("Prophet forecast includes **trend**, **seasonality**, and **holidays** (if configured).")
