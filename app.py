import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="AQI Dashboard", layout="wide")

# ===== Header =====
st.markdown("""
<div class="dashboard-header">
    <h1>AQI Analytics Dashboard</h1>
</div>
""", unsafe_allow_html=True)

# ===== Navigation Bar =====
st.markdown("""
<div class="navbar">
    <a class="nav-link" href="#overview">Overview</a>
    <a class="nav-link" href="#eda">Exploratory Data Analysis</a>
    <a class="nav-link" href="#forecasting">Forecasting</a>
    <a class="nav-link" href="#report">Report</a>
</div>
""", unsafe_allow_html=True)

# Smooth scrolling & active tab highlight
st.markdown("""
<script>
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', function(e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});
</script>
""", unsafe_allow_html=True)

# ===== Load Data =====
@st.cache_data
def load_data():
    df = pd.read_csv("indian_aqi_health_impact_2019_2024.csv")
    df = df.dropna(subset=["City", "AQI"])
    df['Date'] = pd.date_range(start="2019-01-01", periods=len(df), freq='D')[:len(df)]
    return df

df = load_data()

# ===== CSS =====
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ===== AQI Category =====
def aqi_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 200: return "Unhealthy"
    else: return "Very Unhealthy"

# ===== OVERVIEW SECTION =====
st.markdown('<div id="overview"></div>', unsafe_allow_html=True)
st.markdown("<h2 class='section-title'>Overview</h2>", unsafe_allow_html=True)

city_aqi = df.groupby("City")["AQI"].mean().sort_values(ascending=False).reset_index()
most_polluted = city_aqi.iloc[0]
least_polluted = city_aqi.iloc[-1]
avg_aqi = df["AQI"].mean()

# === Card Links ===
most_polluted_url = f"https://www.google.com/search?q={most_polluted['City']}+AQI"
least_polluted_url = f"https://www.google.com/search?q={least_polluted['City']}+AQI"

# === Top Row: Cards ===
col1, col2, col3 = st.columns([1,1,1], gap="large")
with col1:
    st.markdown(f"""
        <a href="{most_polluted_url}" target="_blank" style="text-decoration:none;">
            <div class="card">
                <h3>Most Polluted City</h3>
                <p class="value">{most_polluted['City']}</p>
                <p class="aqi">{most_polluted['AQI']:.1f} ({aqi_category(most_polluted['AQI'])})</p>
            </div>
        </a>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <a href="{least_polluted_url}" target="_blank" style="text-decoration:none;">
            <div class="card">
                <h3>Least Polluted City</h3>
                <p class="value">{least_polluted['City']}</p>
                <p class="aqi">{least_polluted['AQI']:.1f} ({aqi_category(least_polluted['AQI'])})</p>
            </div>
        </a>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="card">
            <h3>Average AQI Nationwide</h3>
            <p class="value">{avg_aqi:.1f}</p>
            <p class="aqi">{aqi_category(avg_aqi)}</p>
        </div>
    """, unsafe_allow_html=True)

# === Bar Charts: Top & Bottom ===
top_10 = city_aqi.head(10)
bottom_10 = city_aqi.tail(10)

custom_colors = ["#7c8c84", "#b49c94", "#9c8c7c", "#3c140c"]

col4, col5 = st.columns(2)
with col4:
    fig_top = px.bar(
        top_10,
        x="AQI", y="City",
        orientation="h",
        title="Top 10 Most Polluted Cities",
        color="AQI",
        color_continuous_scale=custom_colors,
        text=np.round(top_10["AQI"], 1)
    )
    fig_top.update_traces(textposition="outside")
    fig_top.update_layout(plot_bgcolor='#e4dcd4', paper_bgcolor='#e4dcd4')
    st.plotly_chart(fig_top, use_container_width=True)

with col5:
    fig_bottom = px.bar(
        bottom_10,
        x="AQI", y="City",
        orientation="h",
        title="Top 10 Least Polluted Cities",
        color="AQI",
        color_continuous_scale=custom_colors,
        text=np.round(bottom_10["AQI"], 1)
    )
    fig_bottom.update_traces(textposition="outside")
    fig_bottom.update_layout(plot_bgcolor='#e4dcd4', paper_bgcolor='#e4dcd4')
    st.plotly_chart(fig_bottom, use_container_width=True)

# ===== EDA Section =====
st.markdown('<div id="eda" class="section-title">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
st.markdown("""
EDA helps us understand the factors contributing to AQI. We explore which pollutants impact air quality the most,
how weather conditions like temperature, humidity, and wind speed affect AQI, and track AQI changes over time.
""")

# --- 1. Pollutants Impact ---
col1, col2 = st.columns(2)
with col1:
    pollutants = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]
    corr_with_aqi = df[pollutants + ["AQI"]].corr()["AQI"].drop("AQI").sort_values(ascending=False)
    fig_corr = px.bar(
        corr_with_aqi,
        x=corr_with_aqi.values,
        y=corr_with_aqi.index,
        orientation="h",
        title="Which Pollutants Impact AQI the Most?",
        text=np.round(corr_with_aqi.values, 2),
        color=corr_with_aqi.values,
        color_continuous_scale=["#7c8c84", "#b49c94", "#9c8c7c", "#3c140c"]
    )
    fig_corr.update_traces(textposition='outside')
    fig_corr.update_layout(plot_bgcolor='#e4dcd4', paper_bgcolor='#e4dcd4')
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption("PM2.5 and PM10 show the strongest positive correlation with AQI. Lower values (near 0) indicate a weaker relationship.")

# --- 2. Weather Impact on AQI ---
with col2:
    weather_vars = ["Temperature (°C)", "Humidity (%)", "Wind Speed (km/h)", "Rainfall (mm)", "Pressure (hPa)"]
    corr_weather = df[weather_vars + ["AQI"]].corr()["AQI"].drop("AQI").sort_values(ascending=False)
    fig_weather = px.bar(
        corr_weather,
        x=corr_weather.values,
        y=corr_weather.index,
        orientation="h",
        title="How Weather Conditions Affect AQI",
        text=np.round(corr_weather.values, 2),
        color=corr_weather.values,
        color_continuous_scale=["#7c8c84", "#b49c94", "#9c8c7c", "#3c140c"]
    )
    fig_weather.update_traces(textposition='outside')
    fig_weather.update_layout(plot_bgcolor='#e4dcd4', paper_bgcolor='#e4dcd4')
    st.plotly_chart(fig_weather, use_container_width=True)
    st.caption("Weather factors like wind speed and rainfall often reduce AQI by dispersing or washing out pollutants, while stagnant air (low wind) can increase it.")

# --- 3. AQI Trend Over Time ---
aqi_trend = df.groupby("Date")["AQI"].mean().reset_index()
fig_trend = px.line(
    aqi_trend,
    x="Date", y="AQI",
    title="Average AQI Trend Over Time",
    line_shape="spline"
)
fig_trend.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, annotation_text="Good")
fig_trend.add_hrect(y0=51, y1=100, fillcolor="yellow", opacity=0.1, annotation_text="Moderate")
fig_trend.add_hrect(y0=101, y1=200, fillcolor="orange", opacity=0.1, annotation_text="Unhealthy")
fig_trend.add_hrect(y0=201, y1=300, fillcolor="red", opacity=0.1, annotation_text="Very Unhealthy")
fig_trend.update_traces(line=dict(color="#3c140c", width=3))
fig_trend.update_layout(plot_bgcolor='#e4dcd4', paper_bgcolor='#e4dcd4')
st.plotly_chart(fig_trend, use_container_width=True)
st.caption("Notice the seasonal variations: AQI worsens in colder months (possibly due to crop burning and stagnant air) and improves with rainfall.")


# ===== Forecasting Section =====
st.markdown('<div id="forecasting" class="section-title">Forecasting Future AQI Values</div>', unsafe_allow_html=True)

st.markdown("""
<p class="section-intro">
Forecasting helps us anticipate future air quality levels so that citizens and policymakers can take proactive measures. 
Here, you can select a city, adjust the number of forecast days, and pick a start date to see projected AQI levels.
</p>
""", unsafe_allow_html=True)

# === Inputs ===
col_inputs, col_table = st.columns([1,1], gap="large")
with col_inputs:
    city_choice = st.selectbox("Select City for Forecasting", df["City"].unique())
    days = st.slider("Forecast Days", 1, 30, 7)
    start_date = st.date_input("Select Forecast Start Date", df["Date"].max().date())

# === Prepare Data ===
city_data = df[df["City"] == city_choice].groupby("Date")["AQI"].mean().reset_index()
city_data["Days"] = np.arange(len(city_data))
X = city_data[["Days"]]
y = city_data["AQI"]

# Train a simple model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

# Forecast
future_days = np.arange(len(city_data), len(city_data) + days).reshape(-1, 1)
future_aqi = model.predict(future_days)
forecast_dates = pd.date_range(start=start_date, periods=days)
forecast_df = pd.DataFrame({"Date": forecast_dates, "Predicted AQI": np.round(future_aqi, 1)})

# === Top Row: Historical Distribution & Forecast Line Chart ===
col_left, col_right = st.columns([1,2], gap="large")

# Pie chart of historical AQI categories
city_data["Category"] = city_data["AQI"].apply(aqi_category)
category_counts = city_data["Category"].value_counts().reset_index()
category_counts.columns = ["Category", "Count"]

custom_pie_colors = {
    "Good": "#7c8c84", 
    "Moderate": "#b49c94", 
    "Unhealthy": "#9c8c7c", 
    "Very Unhealthy": "#3c140c"
}

fig_pie = px.pie(
    category_counts,
    values="Count",
    names="Category",
    title=f"Historical AQI Categories in {city_choice}",
    color="Category",
    color_discrete_map=custom_pie_colors
)
fig_pie.update_layout(plot_bgcolor='#e4dcd4', paper_bgcolor='#e4dcd4')
col_left.plotly_chart(fig_pie, use_container_width=True)

# Forecast line chart with bands
fig_forecast = px.line(
    forecast_df, x="Date", y="Predicted AQI",
    title=f"Forecasted AQI for {city_choice}",
    line_shape="spline"
)
fig_forecast.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, annotation_text="Good")
fig_forecast.add_hrect(y0=51, y1=100, fillcolor="yellow", opacity=0.1, annotation_text="Moderate")
fig_forecast.add_hrect(y0=101, y1=200, fillcolor="orange", opacity=0.1, annotation_text="Unhealthy")
fig_forecast.add_hrect(y0=201, y1=500, fillcolor="red", opacity=0.1, annotation_text="Very Unhealthy")
fig_forecast.update_traces(line=dict(color="#3c140c", width=3))
fig_forecast.update_layout(plot_bgcolor='#e4dcd4', paper_bgcolor='#e4dcd4')
col_right.plotly_chart(fig_forecast, use_container_width=True)

# === Bottom Row: Forecast Table + Explanation ===
with col_table:
    st.markdown("### Predicted AQI Values")
    st.dataframe(forecast_df, use_container_width=True)

    # Auto explanation
    avg_forecast = forecast_df["Predicted AQI"].mean()
    if avg_forecast > 200:
        reason = "High pollution likely due to winter conditions, crop burning, or heavy traffic congestion."
    elif avg_forecast > 100:
        reason = "Moderate pollution levels expected; local emissions and weather patterns may influence AQI."
    else:
        reason = "Cleaner air expected; favorable weather and reduced pollution sources likely."
    
    st.markdown(f"""
<div class="reason-card small">
    <h4>Why this forecast?</h4>
    <p>{reason}</p>
</div>
""", unsafe_allow_html=True)
    
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
import plotly.io as pio
import io

# ===== Function to Generate Report =====
def generate_report():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Air Quality Index (AQI) Analysis Report", styles['Title']))
    elements.append(Spacer(1, 12))

    # What is AQI
    elements.append(Paragraph("What is AQI?", styles['Heading2']))
    elements.append(Paragraph(
        "The Air Quality Index (AQI) is a standardized measure to assess air pollution levels. "
        "It combines data from pollutants such as PM2.5, PM10, NO2, SO2, CO, and O3 to represent air quality on a scale from 0 (Good) to 500 (Hazardous).",
        styles['Normal']
    ))
    elements.append(Spacer(1, 12))

    # Dataset Info
    elements.append(Paragraph("Dataset Information", styles['Heading2']))
    dataset_url = "https://www.kaggle.com/datasets/divsmahesh/indian-aqi-trends-and-health-risk-20192024"  # <-- Replace with your actual dataset link
    elements.append(Paragraph(
        f"This analysis is based on publicly available air quality data. You can access the dataset here: "
        f"<a href='{dataset_url}' color='blue'>{dataset_url}</a>",
        styles['Normal']
    ))
    elements.append(Spacer(1, 12))

    # Key Insights
    elements.append(Paragraph("Key Insights", styles['Heading2']))
    elements.append(Paragraph(
        f"• The most polluted city: {most_polluted['City']} (AQI: {most_polluted['AQI']:.1f})<br/>"
        f"• The least polluted city: {least_polluted['City']} (AQI: {least_polluted['AQI']:.1f})<br/>"
        f"• Nationwide average AQI: {avg_aqi:.1f}",
        styles['Normal']
    ))
    elements.append(Spacer(1, 12))

    # Save plots as images
    pio.write_image(fig_top, "top_10.png", width=800, height=500)
    pio.write_image(fig_bottom, "bottom_10.png", width=800, height=500)
    pio.write_image(fig_corr, "pollutants_corr.png", width=800, height=500)
    pio.write_image(fig_weather, "weather_corr.png", width=800, height=500)
    pio.write_image(fig_pie, "historical_pie.png", width=800, height=500)
    pio.write_image(fig_forecast, "forecast_chart.png", width=800, height=500)

    # Add Charts
    elements.append(Paragraph("Visual Insights", styles['Heading2']))
    for img_path in ["top_10.png", "bottom_10.png", "pollutants_corr.png", "weather_corr.png", "historical_pie.png", "forecast_chart.png"]:
        elements.append(Image(img_path, width=400, height=250))
        elements.append(Spacer(1, 12))

    # Forecast Table
    elements.append(Paragraph("Forecasted AQI Values", styles['Heading2']))
    table_data = [["Date", "Predicted AQI"]] + forecast_df.values.tolist()
    table = Table(table_data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.black),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 1, colors.grey),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke)
    ]))
    elements.append(table)

    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ===== Report Section in App =====
st.markdown('<div id="report" class="section-title">Download Report</div>', unsafe_allow_html=True)
st.markdown("""
This section provides a professionally generated PDF report summarizing all insights from this dashboard. 
It includes explanations, dataset link, key findings, and visual summaries for easy sharing and reference.
""")

if st.button("Generate & Download Report"):
    pdf_buffer = generate_report()
    st.success("Your report has been generated!")
    st.download_button(
        label="Download Report PDF",
        data=pdf_buffer,
        file_name="AQI_Analysis_Report.pdf",
        mime="application/pdf"
    )