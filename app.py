import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
from io import BytesIO
from pathlib import Path

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Construction Delay Prediction",
    page_icon="🏗️",
    layout="wide"
)

# --------------------------------------------------
# Load Model and Scaler (works from any start folder)
# --------------------------------------------------

BASE_DIR = Path(__file__).parent


@st.cache_resource
def load_artifacts():
    model_ = joblib.load(BASE_DIR / "logistic_delay_model.pkl")
    scaler_ = joblib.load(BASE_DIR / "delay_scaler.pkl")
    return model_, scaler_


model, scaler = load_artifacts()

# Feature names and order used when the model was trained
FEATURE_COLUMNS = [
    "Project_Area_sqft",
    "Number_of_Floors",
    "Planned_Duration_Months",
    "Planned_Manpower",
    "Average_Manpower",
    "Material_Availability_Percent",
    "Equipment_Availability_Percent",
    "Material_Delay_Days",
    "Procurement_Delay_Days",
    "Weather_Delay_Days",
    "Change_Order_Count",
    "Planned_Progress_Percent",
    "Actual_Progress_Percent",
    "Progress_Gap_Percent",
    "Project_Type_Residential",
]

# --------------------------------------------------
# Reset Function
# --------------------------------------------------


def reset_inputs():
    st.session_state["project_type"] = "Residential"
    st.session_state["project_area"] = 2500
    st.session_state["number_of_floors"] = 2
    st.session_state["planned_duration"] = 12
    st.session_state["planned_manpower"] = 30
    st.session_state["average_manpower"] = 25
    st.session_state["material_availability"] = 90.0
    st.session_state["equipment_availability"] = 92.0
    st.session_state["material_delay"] = 4
    st.session_state["procurement_delay"] = 4
    st.session_state["weather_delay"] = 2
    st.session_state["change_orders"] = 2
    st.session_state["planned_progress"] = 60.0
    st.session_state["actual_progress"] = 50.0


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🏗️ Construction Project Delay Prediction")

st.write(
    "AI-Based Construction Project Delay Prediction and Risk Analysis System"
)

# --------------------------------------------------
# About the Project
# --------------------------------------------------

with st.expander("ℹ️ About This Project"):

    st.write(
        """
        This application uses Machine Learning to predict whether a
        construction project is likely to experience a delay.

        The system analyzes project details such as project size,
        manpower, material and equipment availability, delays,
        change orders, and progress information.

        A Logistic Regression model generates the delay prediction
        and probability.

        Based on the predicted probability, the project is categorized
        into Low, Medium, or High Risk.

        Technologies Used:
        • Python
        • Pandas
        • NumPy
        • Scikit-learn
        • Streamlit
        • Joblib
        """
    )

# --------------------------------------------------
# Model Information
# --------------------------------------------------

with st.expander("🤖 Model Information"):

    model_col1, model_col2, model_col3 = st.columns(3)

    with model_col1:
        st.write("**Algorithm**")
        st.write("Logistic Regression")

    with model_col2:
        st.write("**Purpose**")
        st.write("Predict construction project delay")

    with model_col3:
        st.write("**Input Features**")
        st.write("15 project-related features")

    st.write("**Output:** Delay status, delay probability, and risk level.")
    st.write("**Risk Categories:** Low Risk, Medium Risk, High Risk")

# --------------------------------------------------
# Model Performance
# --------------------------------------------------

with st.expander("📊 Model Performance"):

    perf_col1, perf_col2, perf_col3 = st.columns(3)

    with perf_col1:
        st.metric("Accuracy", "75.00%")

    with perf_col2:
        st.metric("Delayed Class F1-Score", "76.00%")

    with perf_col3:
        st.metric("No Delay Class F1-Score", "74.00%")

    st.write(
        "The Logistic Regression model was evaluated using a "
        "stratified train-test split. The test set contained "
        "80 projects."
    )

    st.write(
        "**No Delay (Class 0):** "
        "Precision 76%, Recall 72%, F1-Score 74%"
    )

    st.write(
        "**Delayed (Class 1):** "
        "Precision 74%, Recall 78%, F1-Score 76%"
    )

# --------------------------------------------------
# Project Details
# --------------------------------------------------

st.header("📋 Project Details")

# --------------------------------------------------
# Basic Project Information
# --------------------------------------------------

st.subheader("🏗️ Basic Project Information")

col1, col2 = st.columns(2)

with col1:

    project_type = st.selectbox(
        "Project Type",
        ["Residential", "Commercial"],
        key="project_type"
    )

    project_area = st.number_input(
        "Project Area (sqft)",
        min_value=500,
        max_value=100000,
        value=2500,
        key="project_area"
    )

    number_of_floors = st.number_input(
        "Number of Floors",
        min_value=1,
        max_value=50,
        value=2,
        key="number_of_floors"
    )

with col2:

    planned_duration = st.number_input(
        "Planned Duration (Months)",
        min_value=1,
        max_value=120,
        value=12,
        key="planned_duration"
    )

    planned_manpower = st.number_input(
        "Planned Manpower",
        min_value=1,
        max_value=200,
        value=30,
        key="planned_manpower"
    )

    average_manpower = st.number_input(
        "Average Manpower",
        min_value=1,
        max_value=200,
        value=25,
        key="average_manpower"
    )

if average_manpower > planned_manpower:
    st.warning(
        "Average Manpower is greater than Planned Manpower. "
        "Please check the values."
    )

# --------------------------------------------------
# Resource and Delay Information
# --------------------------------------------------

st.subheader("📦 Resource & Delay Information")

col3, col4 = st.columns(2)

with col3:

    material_availability = st.number_input(
        "Material Availability (%)",
        min_value=0.0,
        max_value=100.0,
        value=90.0,
        key="material_availability"
    )

    equipment_availability = st.number_input(
        "Equipment Availability (%)",
        min_value=0.0,
        max_value=100.0,
        value=92.0,
        key="equipment_availability"
    )

    material_delay = st.number_input(
        "Material Delay (Days)",
        min_value=0,
        max_value=100,
        value=4,
        key="material_delay"
    )

with col4:

    procurement_delay = st.number_input(
        "Procurement Delay (Days)",
        min_value=0,
        max_value=100,
        value=4,
        key="procurement_delay"
    )

    weather_delay = st.number_input(
        "Weather Delay (Days)",
        min_value=0,
        max_value=100,
        value=2,
        key="weather_delay"
    )

    change_orders = st.number_input(
        "Change Order Count",
        min_value=0,
        max_value=50,
        value=2,
        key="change_orders"
    )

# --------------------------------------------------
# Progress Details
# --------------------------------------------------

st.subheader("📊 Progress Details")

col5, col6 = st.columns(2)

with col5:

    planned_progress = st.number_input(
        "Planned Progress (%)",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        key="planned_progress"
    )

with col6:

    actual_progress = st.number_input(
        "Actual Progress (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        key="actual_progress"
    )

# --------------------------------------------------
# Progress Gap
# --------------------------------------------------

progress_gap = planned_progress - actual_progress

if actual_progress > planned_progress:

    st.error(
        "⚠️ Actual Progress cannot be greater than Planned Progress."
    )

else:

    st.info(
        f"📊 Progress Gap: **{progress_gap:.2f}%**"
    )

# --------------------------------------------------
# Buttons
# --------------------------------------------------

st.write("")

button_col1, button_col2 = st.columns([1.3, 1.3])

with button_col1:

    predict_clicked = st.button(
        "🔮 Predict Project Delay",
        type="primary"
    )

with button_col2:

    st.button(
        "🔄 Clear / Reset",
        on_click=reset_inputs
    )

# --------------------------------------------------
# Prediction Processing
# --------------------------------------------------

if predict_clicked:

    # --------------------------------------------------
    # Validate Progress
    # --------------------------------------------------

    if actual_progress > planned_progress:

        st.error(
            "Prediction cannot be performed because Actual Progress "
            "cannot be greater than Planned Progress."
        )

        st.stop()

    # --------------------------------------------------
    # Check Values Against the Training Data Range
    # --------------------------------------------------

    training_ranges = {
        "Project Area": (project_area, 5000, 100000),
        "Number of Floors": (number_of_floors, 1, 20),
        "Planned Duration": (planned_duration, 6, 29),
        "Planned Manpower": (planned_manpower, 10, 76),
        "Average Manpower": (average_manpower, 7, 74),
        "Material Availability": (material_availability, 75.5, 100),
        "Equipment Availability": (equipment_availability, 74.4, 100),
        "Material Delay": (material_delay, 0, 25),
        "Procurement Delay": (procurement_delay, 0, 22),
        "Weather Delay": (weather_delay, 0, 15),
        "Change Orders": (change_orders, 0, 8),
        "Planned Progress": (planned_progress, 15.1, 84.5),
        "Actual Progress": (actual_progress, 5, 82),
        "Progress Gap": (progress_gap, 0, 29.3),
    }

    outside_range = [
        name
        for name, (value, low, high) in training_ranges.items()
        if value < low or value > high
    ]

    # --------------------------------------------------
    # Prepare Input Data
    # --------------------------------------------------

    project_type_residential = (
        1 if project_type == "Residential" else 0
    )

    input_df = pd.DataFrame(
        [[
            project_area,
            number_of_floors,
            planned_duration,
            planned_manpower,
            average_manpower,
            material_availability,
            equipment_availability,
            material_delay,
            procurement_delay,
            weather_delay,
            change_orders,
            planned_progress,
            actual_progress,
            progress_gap,
            project_type_residential
        ]],
        columns=FEATURE_COLUMNS
    )

    # --------------------------------------------------
    # Scale Input
    # --------------------------------------------------

    scaled_data = scaler.transform(input_df)

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    prediction = model.predict(scaled_data)[0]

    probability = (
        model.predict_proba(scaled_data)[0][1] * 100
    )

    # --------------------------------------------------
    # Prediction Text
    # --------------------------------------------------

    if prediction == 1:
        prediction_text = "DELAYED"
    else:
        prediction_text = "ON TIME"

    # --------------------------------------------------
    # Risk Level
    # --------------------------------------------------

    if probability < 40:
        risk_level = "Low Risk 🟢"
    elif probability <= 70:
        risk_level = "Medium Risk 🟡"
    else:
        risk_level = "High Risk 🔴"

    # --------------------------------------------------
    # Prediction Result
    # --------------------------------------------------

    st.divider()

    st.subheader("🎯 Prediction Result")

    if outside_range:
        st.warning(
            "Some entered values are outside the range of the training "
            "data (" + ", ".join(outside_range) + "). "
            "The prediction may be less reliable."
        )

    result_col1, result_col2, result_col3 = st.columns(
        3,
        gap="medium"
    )

    with result_col1:
        st.metric(
            label="Project Status",
            value=prediction_text
        )

    with result_col2:
        st.metric(
            label="Delay Probability",
            value=f"{probability:.2f}%"
        )

    with result_col3:
        st.metric(
            label="Risk Level",
            value=risk_level
        )

    # --------------------------------------------------
    # Risk Level Explanation
    # --------------------------------------------------

    if probability < 40:

        st.success(
            "🟢 Low Risk: The project has a relatively low "
            "probability of delay."
        )

    elif probability <= 70:

        st.warning(
            "🟡 Medium Risk: The project has a moderate "
            "probability of delay. Project factors should "
            "be monitored carefully."
        )

    else:

        st.error(
            "🔴 High Risk: The project has a high probability "
            "of delay. Important delay factors should be reviewed."
        )

    # --------------------------------------------------
    # Probability Visualization
    # --------------------------------------------------

    st.subheader("📈 Delay Probability Visualization")

    st.progress(
        min(int(probability), 100)
    )

    st.write(
        f"Delay Probability: **{probability:.2f}%**"
    )

    # --------------------------------------------------
    # Key Project Indicators
    # --------------------------------------------------

    st.subheader("🔎 Key Project Indicators")

    indicator_col1, indicator_col2, indicator_col3 = st.columns(3)

    with indicator_col1:
        st.metric("Progress Gap", f"{progress_gap:.1f}%")

    with indicator_col2:
        st.metric("Procurement Delay", f"{procurement_delay} days")

    with indicator_col3:
        st.metric("Material Availability", f"{material_availability:.1f}%")

    # --------------------------------------------------
    # Project Summary
    # --------------------------------------------------

    st.subheader("📋 Project Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.write(f"**Project Type:** {project_type}")
        st.write(f"**Project Area:** {project_area:.0f} sqft")
        st.write(f"**Number of Floors:** {number_of_floors}")
        st.write(f"**Planned Duration:** {planned_duration} months")
        st.write(f"**Planned Manpower:** {planned_manpower}")

    with summary_col2:
        st.write(f"**Average Manpower:** {average_manpower}")
        st.write(f"**Material Availability:** {material_availability:.1f}%")
        st.write(f"**Equipment Availability:** {equipment_availability:.1f}%")
        st.write(f"**Material Delay:** {material_delay} days")
        st.write(f"**Procurement Delay:** {procurement_delay} days")

    with summary_col3:
        st.write(f"**Weather Delay:** {weather_delay} days")
        st.write(f"**Change Orders:** {change_orders}")
        st.write(f"**Planned Progress:** {planned_progress:.1f}%")
        st.write(f"**Actual Progress:** {actual_progress:.1f}%")
        st.write(f"**Progress Gap:** {progress_gap:.1f}%")

    # --------------------------------------------------
    # Prediction Interpretation
    # --------------------------------------------------

    st.subheader("📝 Prediction Interpretation")

    if prediction == 1:

        st.warning(
            "The project has a higher likelihood of experiencing "
            "delay. Review manpower, material availability, "
            "procurement, weather, and progress-related factors."
        )

    else:

        st.info(
            "The project has a lower likelihood of experiencing "
            "delay based on the entered project details."
        )

    # --------------------------------------------------
    # Prediction Timestamp
    # --------------------------------------------------

    prediction_time = datetime.now().strftime(
        "%d-%m-%Y %I:%M %p"
    )

    st.caption(
        f"Prediction generated on: {prediction_time}"
    )

    # --------------------------------------------------
    # Prediction Report Data
    # --------------------------------------------------

    report_data = {
        "Project Type": project_type,
        "Project Area (sqft)": project_area,
        "Number of Floors": number_of_floors,
        "Planned Duration (Months)": planned_duration,
        "Planned Manpower": planned_manpower,
        "Average Manpower": average_manpower,
        "Material Availability (%)": material_availability,
        "Equipment Availability (%)": equipment_availability,
        "Material Delay (Days)": material_delay,
        "Procurement Delay (Days)": procurement_delay,
        "Weather Delay (Days)": weather_delay,
        "Change Order Count": change_orders,
        "Planned Progress (%)": planned_progress,
        "Actual Progress (%)": actual_progress,
        "Progress Gap (%)": progress_gap,
        "Prediction": prediction_text,
        "Delay Probability (%)": round(probability, 2),
        "Risk Level": risk_level,
        "Prediction Generated": prediction_time
    }

    report_df = pd.DataFrame([report_data])

    # --------------------------------------------------
    # Download Prediction Report
    # --------------------------------------------------

    st.subheader("📥 Download Prediction Report")

    download_col1, download_col2 = st.columns(2)

    with download_col1:

        st.download_button(
            label="📄 Download CSV",
            data=report_df.to_csv(index=False),
            file_name="construction_delay_prediction_report.csv",
            mime="text/csv"
        )

    with download_col2:

        excel_buffer = BytesIO()

        with pd.ExcelWriter(
            excel_buffer,
            engine="openpyxl"
        ) as writer:

            report_df.to_excel(
                writer,
                index=False,
                sheet_name="Prediction Report"
            )

        st.download_button(
            label="📊 Download Excel",
            data=excel_buffer.getvalue(),
            file_name="construction_delay_prediction_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "AI-Based Construction Project Delay Prediction and Risk Analysis System"
)

st.caption(
    "Developed as an MCA Mini Project using Machine Learning and Streamlit"
)