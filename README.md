# 📊 Unemployment Analysis with Python

This project analyzes and visualizes unemployment rate data using Python. It includes:
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Impact analysis of COVID-19 on unemployment
- Regional and seasonal trend analysis
- Forecasting using Prophet and Random Forest
- Interactive dashboard built with Streamlit
- Animated choropleth map and bar chart visualizations

---

## 📁 Project Structure

unemployment_analysis/
│
├──unemployment_data.csv # Raw dataset
├── rf_model.pkl # Trained Random Forest model
├── forecasting_script.py # Model training script
├── unemployment_analysis.ipynb # Jupyter Notebook for EDA & modeling
├── app.py # Streamlit frontend
├── summary.txt # Key insights
├── requirements.txt # Python dependencies
└── README.md # Project documentation


## 📦 Setup Instructions

1. **Clone the repository**

git clone https://github.com/vardhman18/unemployment-analysis.git

cd unemployment_analysis

2. **Install dependencies**


pip install -r requirements.txt

3. **Run the Streamlit app**


cd streamlit_app

streamlit run app.py

-----


### 📊 Features

 ## ✅ Exploratory Data Analysis
Line charts showing unemployment trends over time

Regional boxplots

Monthly seasonality plots

## 🦠 COVID-19 Impact Analysis
Compare average unemployment rates before and after March 2020

## 🌍 Animated Choropleth Map
Shows unemployment rate by Indian regions over time

## 📈 Forecasting
Predict unemployment for the next 12 months

Choose between:

Prophet (time series)

Random Forest (regressor)

## 📊 Bar Graph
View region-wise unemployment on a specific date

### 🌐 Streamlit Dashboard Preview
Feature	Description
🗺️ Choropleth Map	Animated map of unemployment by region over time
📊 Bar Graph	Date-wise unemployment by region
🔮 Forecasting	Predict future unemployment trends
📅 Monthly Trends	Average unemployment by month
🧪 COVID Comparison	Pre- vs Post-COVID unemployment stats

### 📚 Dataset
The dataset (unemployment_data.csv) contains:

> Date
>Region
>Estimated Unemployment Rate (%)
>Estimated Employed
>Estimated Labour Participation Rate (%)
>Location coordinates (latitude, longitude)

Ensure the column 'Estimated Unemployment Rate (%)' is renamed to 'Unemployment Rate' for compatibility.

### 🤖 Models Used
Prophet: For trend and seasonality-based time series forecasting

Random Forest Regressor: For machine learning-based prediction



