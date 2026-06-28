import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Traffic Demand Prediction Dashboard",
    page_icon="🚦",
    layout="wide"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

[data-testid="stMetric"]{
    background:#111827;
    border:1px solid #2c2c2c;
    border-radius:15px;
    padding:20px;
    box-shadow:0px 3px 8px rgba(0,0,0,0.25);
}

h1{
    color:#00C2FF;
}

hr{
    margin-top:30px;
    margin-bottom:30px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("traffic_model.pkl")

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")

# ----------------------------
# Header
# ----------------------------
st.title("🚦 Traffic Demand Prediction Dashboard")

st.markdown("""
### AI-Powered Traffic Volume Prediction using LightGBM Machine Learning

Predict hourly traffic congestion using weather conditions and time-based features.
""")

st.divider()

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("⚙ Prediction Panel")

st.sidebar.markdown(
"""
Adjust the traffic conditions below
and click **Predict Traffic**.
"""
)

temperature = st.sidebar.number_input(
    "🌡 Temperature (Kelvin)",
    value=288.0
)

rain = st.sidebar.number_input(
    "🌧 Rainfall (1 Hour)",
    value=0.0
)

snow = st.sidebar.number_input(
    "❄ Snowfall (1 Hour)",
    value=0.0
)

clouds = st.sidebar.slider(
    "☁ Cloud Coverage (%)",
    0,
    100,
    50
)

hour = st.sidebar.slider(
    "🕒 Hour",
    0,
    23,
    12
)

day = st.sidebar.slider(
    "📅 Day of Week",
    0,
    6,
    2
)

predict = st.sidebar.button("🚀 Predict Traffic")

# ----------------------------
# Dashboard Metrics
# ----------------------------
c1,c2,c3 = st.columns(3)

with c1:
    st.metric(
        "📄 Dataset Records",
        f"{df.shape[0]:,}"
    )

with c2:
    st.metric(
        "📑 Features",
        df.shape[1]
    )

with c3:
    st.metric(
        "❗ Missing Cells",
        int(df.isnull().sum().sum())
    )

# ----------------------------
# Model Information
# ----------------------------
st.info("""
### 🤖 Model Information

**Model Used:** LightGBM Regressor

**Dataset:** Metro Interstate Traffic Volume

**Records:** 48,204

**Features:** 14

**Target Variable:** Traffic Volume
""")

st.success("""
### 📊 Model Performance

✅ **R² Score:** 0.9748

✅ **MAE:** 198.63

✅ **RMSE:** 315.60
""")

with st.expander("📂 View Dataset Preview"):
    st.dataframe(df.head(10))

# ==========================================================
# Prediction Section
# ==========================================================

if predict:

    rush_hour = 1 if hour in [7, 8, 9, 16, 17, 18] else 0
    weekend = 1 if day >= 5 else 0

    input_data = pd.DataFrame([[
        0,                  # holiday
        temperature,
        rain,
        snow,
        clouds,
        1,                  # weather_main
        1,                  # weather_description
        2018,
        6,
        15,
        hour,
        day,
        rush_hour,
        weekend
    ]], columns=[
        "holiday",
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "weather_main",
        "weather_description",
        "year",
        "month",
        "day",
        "hour",
        "dayofweek",
        "rush_hour",
        "weekend"
    ])

    prediction = model.predict(input_data)[0]

    st.divider()

    st.markdown("## 📈 Prediction Result")

    c1, c2 = st.columns([2,1])

    with c1:

        st.metric(
            label="🚗 Predicted Traffic Volume",
            value=f"{int(prediction):,}",
            delta="Vehicles / Hour"
        )

    with c2:

        if prediction < 2000:
            st.success("🟢 Low Traffic")

        elif prediction < 4000:
            st.warning("🟡 Moderate Traffic")

        else:
            st.error("🔴 Heavy Traffic")

    st.write("")

    if rush_hour:
        st.info("⏰ Rush Hour : YES")
    else:
        st.success("🌙 Rush Hour : NO")

    st.write("")

    st.markdown("### 💡 Travel Recommendation")

    if prediction > 4000:

        st.error(
            """
Heavy congestion is expected.

• Consider using alternate routes.

• Start your journey earlier.

• Avoid peak traffic hours if possible.
"""
        )

    elif prediction > 2500:

        st.warning(
            """
Moderate traffic is expected.

• Plan an extra 10–15 minutes.

• Expect slower movement near junctions.
"""
        )

    else:

        st.success(
            """
Traffic conditions are smooth.

• Normal travel conditions.

• Safe to proceed with your journey.
"""
        )

# ==========================================================
# Dashboard Analytics
# ==========================================================

st.divider()

st.markdown("## 📊 Traffic Dataset Analytics")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Average Traffic",
        f"{int(df['traffic_volume'].mean()):,}"
    )

with c2:
    st.metric(
        "Maximum Traffic",
        f"{int(df['traffic_volume'].max()):,}"
    )

with c3:
    st.metric(
        "Minimum Traffic",
        f"{int(df['traffic_volume'].min()):,}"
    )

st.write("")

# ==========================================================
# Traffic Trend
# ==========================================================

st.markdown("## 📈 Traffic Volume Trend")

# Create Hour Column
df["date_time"] = pd.to_datetime(df["date_time"])

df["hour"] = df["date_time"].dt.hour

traffic_hour = (
    df.groupby("hour")["traffic_volume"]
      .mean()
      .reset_index()
)

fig, ax = plt.subplots(figsize=(10,4))

ax.plot(
    traffic_hour["hour"],
    traffic_hour["traffic_volume"],
    linewidth=3
)

ax.set_title("Average Traffic Volume by Hour")

ax.set_xlabel("Hour of Day")

ax.set_ylabel("Traffic Volume")

ax.grid(alpha=0.3)

st.pyplot(fig) 

# ==========================================================
# Footer
# ==========================================================

st.divider()

st.markdown(
"""
### 🚀 Project Summary

This dashboard predicts hourly traffic volume using a trained
**LightGBM Machine Learning model**.

The prediction considers weather conditions,
cloud coverage,
temperature,
rainfall,
snowfall,
hour of the day,
day of the week,
rush-hour status,
and weekend information.

The dashboard was developed using:

- 🐍 Python
- 🚀 Streamlit
- 🌳 LightGBM
- 📊 Pandas
- 🤖 Scikit-Learn
"""
)

st.caption(
"""
Developed for the Innovex Machine Learning Internship Program

Traffic Demand Prediction using Machine Learning
"""
)