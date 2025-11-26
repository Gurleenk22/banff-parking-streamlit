import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import PartialDependenceDisplay

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Banff Parking – ML & XAI Dashboard",
    layout="wide"
)

st.write("")  # small spacing

# ---------------------------------------------------
# LOAD MODELS + DATA (CACHED)
# ---------------------------------------------------
@st.cache_resource
def load_models_and_data():
    # These filenames must match what you uploaded to GitHub
    reg = joblib.load("banff_best_xgb_reg.pkl")      # XGBoost regressor
    cls = joblib.load("banff_best_xgb_cls.pkl")      # XGBoost classifier
    scaler = joblib.load("banff_scaler.pkl")         # Scaler used in training
    features = joblib.load("banff_features.pkl")     # List of feature names

    # Test data for XAI and residual analysis
    X_test_scaled = np.load("X_test_scaled.npy")
    y_reg_test = np.load("y_reg_test.npy")

    return reg, cls, scaler, features, X_test_scaled, y_reg_test


best_xgb_reg, best_xgb_cls, scaler, FEATURES, X_test_scaled, y_reg_test = load_models_and_data()

# ---------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Make Prediction", "XAI – Explainable AI"]
)

# ---------------------------------------------------
# PAGE 1 – OVERVIEW
# ---------------------------------------------------
if page == "Overview":
    st.title("Banff Parking Demand – Machine Learning Project")

    st.markdown(
        """
        This dashboard is built from a Banff parking analytics project.

        **Goals of the project**

        - Understand how **time**, **weather**, and **historical occupancy**
          affect hourly parking demand.
        - Predict **hourly occupancy** for Banff parking lots.
        - Estimate the **probability that a lot is near full** (e.g., > 90%).
        - Use **Explainable AI (XAI)** to show which features drive the model.

        **Data sources used in the project**

        - Parking management data (transactions, stalls, units/lots).
        - Visits / routes data to understand traffic arriving into Banff.
        - Enriched weather and time-based features (month, day-of-week, hour,
          weekend/weekday, lag occupancy, rolling averages).
        """
    )

    st.info(
        "Use the menu on the left to switch between: "
        "**Overview**, **Make Prediction**, and **XAI – Explainable AI**."
    )

# ---------------------------------------------------
# PAGE 2 – MAKE PREDICTION
# ---------------------------------------------------
if page == "Make Prediction":
    st.title("Predict Parking Occupancy & Full-Lot Risk")

    st.markdown(
        """
        Use this page to simulate a future hour and see:

        - Predicted **occupancy level** (regression model)
        - **Probability the lot is full / near capacity** (classification model)
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        month = st.slider("Month (1 = Jan, 12 = Dec)", 1, 12, 7)
        day_of_week = st.slider("Day of Week (0 = Monday, 6 = Sunday)", 0, 6, 5)
        hour = st.slider("Hour of Day (0–23)", 0, 23, 14)

    with col2:
        max_temp = st.slider("Max Temperature (°C)", -20.0, 40.0, 22.0)
        total_precip = st.slider("Total Precipitation (mm)", 0.0, 30.0, 0.5)
        wind_gust = st.slider("Speed of Max Gust (km/h)", 0.0, 100.0, 15.0)

    is_weekend = 1 if day_of_week in [5, 6] else 0

    st.caption(
        "Note: Lag features (e.g., previous-hour occupancy) are set to 0 in this demo "
        "unless they are explicitly provided in the feature list."
    )

    # Build a feature dictionary based on the most important columns
    input_dict = {
        "Month": month,
        "DayOfWeek": day_of_week,
        "Hour": hour,
        "IsWeekend": is_weekend,
        "Max Temp (°C)": max_temp,
        "Total Precip (mm)": total_precip,
        "Spd of Max Gust (km/h)": wind_gust,
        # Any remaining features not listed here will default to 0
    }

    # Align to the exact FEATURE order used during training
    x_vector = np.array([input_dict.get(f, 0) for f in FEATURES]).reshape(1, -1)

    # Scale with the same scaler used in training
    x_scaled = scaler.transform(x_vector)

    if st.button("Predict"):
        # Regression: occupancy
        occ_pred = best_xgb_reg.predict(x_scaled)[0]

        # Classification: probability of full / near capacity
        full_prob = best_xgb_cls.predict_proba(x_scaled)[0, 1]

        st.subheader("Prediction Results")

        st.write(f"**Predicted Occupancy (model units):** `{occ_pred:.2f}`")
        st.write(f"**Probability Lot is Full / Near Capacity:** `{full_prob:.1%}`")

        if full_prob > 0.7:
            st.warning(
                "High risk that this lot will be full. Consider re-directing drivers "
                "to alternative parking or adjusting signage."
            )
        elif full_prob > 0.4:
            st.info(
                "Moderate risk of the lot being busy. Monitoring and dynamic wayfinding "
                "may be helpful."
            )
        else:
            st.success(
                "Low risk of the lot being at full capacity for this hour."
            )

# ---------------------------------------------------
# PAGE 3 – XAI (EXPLAINABLE AI)
# ---------------------------------------------------
if page == "XAI – Explainable AI":
    st.title("Explainable AI – Understanding the Models")

    st.markdown(
        """
        This page explains **why** the models make their predictions,
        using Explainable AI tools:

        - **SHAP summary plot**: which features contribute most to predictions.
        - **SHAP bar plot**: global feature importance.
        - **Partial Dependence Plots (PDPs)**: how changing one feature
          affects predicted occupancy.
        - **Residual plot**: how close predictions are to the true values.
        """
    )

    # ---------- SHAP EXPLANATIONS FOR REGRESSION ----------
    st.subheader("SHAP Summary – Regression Model (Occupancy)")

    try:
        explainer_reg = shap.TreeExplainer(best_xgb_reg)
        shap_values_reg = explainer_reg.shap_values(X_test_scaled)

        # Summary dot plot
        fig1, ax1 = plt.subplots()
        shap.summary_plot(
            shap_values_reg,
            X_test_scaled,
            feature_names=FEATURES,
            show=False
        )
        st.pyplot(fig1)
        st.caption(
            "Each point represents a sample. Colour shows feature value, and position "
            "shows how much that feature pushed the prediction up or down."
        )

        # Summary bar plot
        st.subheader("SHAP Feature Importance – Regression")
        fig2, ax2 = plt.subplots()
        shap.summary_plot(
            shap_values_reg,
            X_test_scaled,
            feature_names=FEATURES,
            plot_type="bar",
            show=False
        )
        st.pyplot(fig2)
    except Exception as e:
        st.error(f"Could not generate SHAP plots: {e}")

    # ---------- PARTIAL DEPENDENCE PLOTS ----------
    st.subheader("Partial Dependence – Key Features")

    # Choose some typical features; change names if your features differ
    pd_feature_names = []
    for name in ["Max Temp (°C)", "Month", "Hour"]:
        if name in FEATURES:
            pd_feature_names.append(name)

    if len(pd_feature_names) > 0:
        feature_indices = [FEATURES.index(f) for f in pd_feature_names]
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        PartialDependenceDisplay.from_estimator(
            best_xgb_reg,
            X_test_scaled,
            feature_indices,
            feature_names=FEATURES,
            ax=ax3
        )
        st.pyplot(fig3)
        st.caption(
            "Partial dependence shows the average effect of each feature on predicted "
            "occupancy while holding other features constant."
        )
    else:
        st.info(
            "Could not find the configured PDP features ('Max Temp (°C)', 'Month', 'Hour') "
            "in the FEATURES list. You may need to adjust the feature names."
        )

    # ---------- RESIDUAL ANALYSIS ----------
    st.subheader("Residual Plot – Regression Model")

    try:
        y_pred = best_xgb_reg.predict(X_test_scaled)
        residuals = y_reg_test - y_pred

        fig4, ax4 = plt.subplots()
        ax4.scatter(y_pred, residuals, alpha=0.3)
        ax4.axhline(0, color="red", linestyle="--")
        ax4.set_xlabel("Predicted Occupancy")
        ax4.set_ylabel("Residual (Actual - Predicted)")
        st.pyplot(fig4)
        st.caption(
            "Residuals scattered symmetrically around zero suggest that the model "
            "captures the main patterns without strong systematic bias."
        )
    except Exception as e:
        st.error(f"Could not compute residuals: {e}")

