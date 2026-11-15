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
