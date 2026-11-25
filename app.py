
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from datetime import date

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Bangalore Traffic Prediction",
    page_icon="🚦",
    layout="wide"
)

# --------------------------------
# LOAD TRAINED MODEL
# --------------------------------

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/best_traffic_model.pkl"
    )

    preprocessor = joblib.load(
        "models/traffic_preprocessor.pkl"
    )

    label_encoder = joblib.load(
        "models/label_encoder.pkl"
    )

    features = joblib.load(
        "models/feature_columns.pkl"
    )

    return model, preprocessor, label_encoder, features


# --------------------------------
# LOAD DATASET
# --------------------------------

@st.cache_data
def load_dataset():

    return pd.read_csv(
        "Banglore_traffic_Dataset.csv"
    )


model, preprocessor, label_encoder, features = load_model()

data = load_dataset()


# --------------------------------
# APPLICATION HEADER
# --------------------------------

st.title("🚦 Bangalore Traffic Congestion Prediction")

st.write(
    "Machine Learning Based Traffic Congestion "
    "Classification and Hotspot Analysis"
)

st.divider()

st.subheader("Enter Traffic Information")


# --------------------------------
# USER INPUT FORM
# --------------------------------

with st.form("traffic_prediction_form"):

    col1, col2 = st.columns(2)

    # Area selection

    with col1:

        area = st.selectbox(
            "Select Area",
            sorted(data["Area Name"].dropna().unique())
        )

        road = st.selectbox(
            "Select Road / Intersection",
            sorted(
                data["Road/Intersection Name"]
                .dropna()
                .unique()
            )
        )

        weather = st.selectbox(
            "Weather Conditions",
            sorted(
                data["Weather Conditions"]
                .dropna()
                .unique()
            )
        )

    with col2:

        roadwork = st.selectbox(
            "Roadwork Activity",
            sorted(
                data["Roadwork and Construction Activity"]
                .dropna()
                .unique()
            )
        )

        selected_date = st.date_input(
            "Select Observation Date",
            value=date(2024, 8, 1)
        )

    st.divider()

    st.subheader("Additional Traffic Information")

    numeric_features = [
        "Incident Reports",
        "Traffic Signal Compliance",
        "Public Transport Usage",
        "Parking Usage",
        "Pedestrian and Cyclist Count"
    ]

    numeric_values = {}

    col3, col4 = st.columns(2)

    for index, feature in enumerate(numeric_features):

        column = col3 if index % 2 == 0 else col4

        with column:

            median_value = float(
                pd.to_numeric(data[feature], errors="coerce").median()
            )

            numeric_values[feature] = st.number_input(
                feature,
                value=median_value
            )

    # Submit button

    submit = st.form_submit_button(
        "Predict Traffic Congestion",
        use_container_width=True
    )


# --------------------------------
# PREDICTION
# --------------------------------

if submit:

    # Extract date features

    year = selected_date.year

    month = selected_date.month

    day = selected_date.day

    day_of_week = selected_date.weekday()

    is_weekend = int(day_of_week >= 5)

    # Season function

    if month in [12, 1, 2]:

        season = "Winter"

    elif month in [3, 4, 5]:

        season = "Summer"

    elif month in [6, 7, 8, 9]:

        season = "Monsoon"

    else:

        season = "Post-Monsoon"

    # Create input dictionary

    input_data = {

        "Year": year,

        "Month": month,

        "Day": day,

        "Day_of_Week": day_of_week,

        "Is_Weekend": is_weekend,

        "Incident Reports": numeric_values["Incident Reports"],

        "Traffic Signal Compliance": numeric_values["Traffic Signal Compliance"],

        "Public Transport Usage": numeric_values["Public Transport Usage"],

        "Parking Usage": numeric_values["Parking Usage"],

        "Pedestrian and Cyclist Count": numeric_values["Pedestrian and Cyclist Count"],

        "Area Name": area,

        "Road/Intersection Name": road,

        "Weather Conditions": weather,

        "Roadwork and Construction Activity": roadwork,

        "Season": season

    }

    # Convert dictionary into DataFrame

    input_df = pd.DataFrame(
        [input_data]
    )

    # Arrange columns in training order

    input_df = input_df[features]

    # Preprocess input

    processed_data = preprocessor.transform(
        input_df
    )

    # Predict congestion

    prediction = model.predict(
        processed_data
    )

    # Convert numerical prediction into text

    result = label_encoder.inverse_transform(
        prediction
    )[0]

    # Display prediction

    st.divider()

    st.subheader("Prediction Result")

    if result == "High":

        st.error("HIGH TRAFFIC CONGESTION")

    elif result == "Medium":

        st.warning("MEDIUM TRAFFIC CONGESTION")

    else:

        st.success("LOW TRAFFIC CONGESTION")

    st.write("Predicted Category:", result)

    # Show submitted input

    with st.expander("View Submitted Traffic Information"):

        st.dataframe(
            input_df,
            use_container_width=True
        )

    st.caption(
        "This application classifies congestion from supplied "
        "traffic conditions. It does not forecast future traffic."
    )
