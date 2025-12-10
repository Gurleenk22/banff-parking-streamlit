import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.inspection import PartialDependenceDisplay

# ---------------------------------------------------
# BASIC PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Banff Parking – ML & XAI Dashboard",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODELS + DATA (CACHED)
# ---------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """
    Load trained models, scaler, feature list, and test data.
    Filenames must match those in the GitHub repo.
    """
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
st.sidebar.title("Banff Parking Dashboard")
st.sidebar.markdown(
    """
    Use this app to:
    - Explore hourly parking demand  
    - Check which lots may be full  
    - Understand the model using XAI  
    """
)

page = st.sidebar.radio(
    "Go to",
    ["Overview", "Make Prediction", "Lot Status Overview", "XAI – Explainable AI"]
)

# ---------------------------------------------------
# PAGE 1 – OVERVIEW (MORE ATTRACTIVE)
# ---------------------------------------------------
if page == "Overview":
    st.title("🚗 Banff Parking Demand – Machine Learning Overview")

    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.markdown(
            """
            ### Project Question

            *How can Banff use real data to anticipate parking pressure and avoid full lots during the May–September tourist season?*

            This project combines:
  
            - *Parking management data* – when and where people park  
            - *Weather data* – temperature, rain, and wind  
            - *Engineered features* – hour, weekday/weekend, lagged occupancy, rolling averages  

            A Gradient-boosted tree model (*XGBoost*) predicts:
            - Hourly *occupancy level* for each lot  
            - *Probability that a lot is near full* (> 90% capacity)  
            """
        )

    with col_right:
        st.markdown("### Quick Facts (from engineered data)")
        kpi1, kpi2 = st.columns(2)
        with kpi1:
            st.metric("Tourist season", "May–September 2025")
        with kpi2:
            st.metric("Lots modelled", "Multiple Banff units")
        kpi3, kpi4 = st.columns(2)
        with kpi3:
            st.metric("Target 1", "Hourly occupancy")
        with kpi4:
            st.metric("Target 2", "Full / Not-full")

        st.markdown(
            """
            ✅ Models trained on *historical hourly data*  
            ✅ Includes *time, weather, and history* features  
            ✅ Deployed as this *Streamlit decision-support app*
            """
        )

    st.markdown("---")

    st.subheader("How to Use This App")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            *1. Make Prediction*  
            - Choose a *lot & scenario*  
            - Adjust *time & weather*  
            - See:  
              - Predicted occupancy  
              - Probability the lot is full  
              - Trend for the *next 6 hours*
            """
        )

    with col2:
        st.markdown(
            """
            *2. Lot Status Overview*  
            - Select a *single hour*  
            - Compare *all lots*  
            - Status: 🟥 High risk full, 🟧 Busy, 🟩 Comfortable  
            - Supports operational decisions & signage
            """
        )

    with col3:
        st.markdown(
            """
            *3. XAI – Explainable AI*  
            - Global *SHAP* feature importance  
            - *Partial Dependence Plots* (Hour, Month, Temp)  
            - *Residual plot* to check model fit  
            - Helps justify decisions to stakeholders
            """
        )

    st.info(
        "Tip: move between pages using the left sidebar. Start with "
        "*Make Prediction* to see how the model behaves for different scenarios."
    )

# ---------------------------------------------------
# PAGE 2 – MAKE PREDICTION (IMPROVED & INTERACTIVE)
# ---------------------------------------------------
if page == "Make Prediction":
    st.title("🎯 Interactive Parking Demand Prediction")

    st.markdown(
        """
        Use this page to explore what-if scenarios for a single Banff parking lot.

        1. Select a *parking lot*  
        2. Choose a *scenario* (or adjust the sliders)  
        3. See:
           - Predicted *occupancy* for the selected hour  
           - *Probability* the lot is near full  
           - A *trend for the next 6 hours* keeping weather & day constant
        """
    )

    # Find lot indicator features (one-hot encoded units)
    lot_features = [f for f in FEATURES if f.startswith("Unit_")]
    lot_display_names = [lf.replace("Unit_", "").replace("_", " ") for lf in lot_features]

    if not lot_features:
        st.warning(
            "No parking-lot indicator features (starting with 'Unit_') were "
            "found in FEATURES. Lot selection is disabled; generic features only."
        )

    # Scenario presets
    scenario_options = {
        "Custom (use sliders below)": None,
        "Sunny Weekend Midday": {"month": 7, "dow": 5, "hour": 13,
                                 "max_temp": 24.0, "precip": 0.0, "gust": 10.0},
        "Rainy Weekday Afternoon": {"month": 6, "dow": 2, "hour": 16,
                                    "max_temp": 15.0, "precip": 5.0, "gust": 20.0},
        "Cold Morning (Shoulder Season)": {"month": 5, "dow": 1, "hour": 9,
                                           "max_temp": 5.0, "precip": 0.0, "gust": 15.0},
        "Warm Evening (Busy Day)": {"month": 8, "dow": 6, "hour": 19,
                                    "max_temp": 22.0, "precip": 0.0, "gust": 8.0},
    }

    st.subheader("Step 1 – Choose Lot & Scenario")

    col_lot, col_scenario = st.columns([1.2, 1])

    with col_lot:
        if lot_features:
            selected_lot_label = st.selectbox(
                "Select parking lot",
                lot_display_names,
                index=0
            )
            selected_lot_feature = lot_features[lot_display_names.index(selected_lot_label)]
        else:
            selected_lot_label = None
            selected_lot_feature = None

    with col_scenario:
        selected_scenario = st.selectbox(
            "Scenario",
            list(scenario_options.keys()),
            index=1
        )

    # Default slider values – will be overwritten by scenario if chosen
    default_vals = {"month": 7, "dow": 5, "hour": 13,
                    "max_temp": 22.0, "precip": 0.5, "gust": 12.0}

    if scenario_options[selected_scenario] is not None:
        default_vals.update(scenario_options[selected_scenario])

    st.subheader("Step 2 – Adjust Conditions (if needed)")

    col1, col2 = st.columns(2)

    with col1:
        month = st.slider("Month (1 = Jan, 12 = Dec)",
                          1, 12, int(default_vals["month"]))
        day_of_week = st.slider("Day of Week (0 = Monday, 6 = Sunday)",
                                0, 6, int(default_vals["dow"]))
        hour = st.slider("Hour of Day (0–23)",
                         0, 23, int(default_vals["hour"]))

    with col2:
        max_temp = st.slider("Max Temperature (°C)",
                             -20.0, 40.0, float(default_vals["max_temp"]))
        total_precip = st.slider("Total Precipitation (mm)",
                                 0.0, 30.0, float(default_vals["precip"]))
        wind_gust = st.slider("Speed of Max Gust (km/h)",
                              0.0, 100.0, float(default_vals["gust"]))

    is_weekend = 1 if day_of_week in [5, 6] else 0

    st.caption(
        "Lag features (previous-hour occupancy, rolling averages) are not entered manually. "
        "For the 6-hour forecast we create a simple synthetic history using the model’s "
        "own predictions."
    )

    # Build feature dict starting from all zeros
    base_input = {f: 0 for f in FEATURES}

    # Time & weather
    if "Month" in base_input:
        base_input["Month"] = month
    if "DayOfWeek" in base_input:
        base_input["DayOfWeek"] = day_of_week
    if "Hour" in base_input:
        base_input["Hour"] = hour
    if "IsWeekend" in base_input:
        base_input["IsWeekend"] = is_weekend
    if "Max Temp (°C)" in base_input:
        base_input["Max Temp (°C)"] = max_temp
    if "Total Precip (mm)" in base_input:
        base_input["Total Precip (mm)"] = total_precip
    if "Spd of Max Gust (km/h)" in base_input:
        base_input["Spd of Max Gust (km/h)"] = wind_gust

    # Lot indicator – one-hot
    if selected_lot_feature is not None and selected_lot_feature in base_input:
        base_input[selected_lot_feature] = 1

    # Vector in the exact training feature order
    x_vec = np.array([base_input[f] for f in FEATURES]).reshape(1, -1)
    x_scaled = scaler.transform(x_vec)

    if st.button("🔮 Predict for this scenario"):
        # Current-hour predictions
        occ_pred = best_xgb_reg.predict(x_scaled)[0]
        full_prob = best_xgb_cls.predict_proba(x_scaled)[0, 1]

        st.subheader("Step 3 – Results for Selected Hour")

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric("Predicted occupancy (model units)",
                      f"{occ_pred:.2f}")
        with col_res2:
            st.metric("Probability lot is near full",
                      f"{full_prob:.1%}")

        if full_prob > 0.7:
            st.warning(
                "⚠ High risk this lot will be full. Consider redirecting drivers "
                "to other parking areas or adjusting signage."
            )
        elif full_prob > 0.4:
            st.info(
                "Moderate risk of heavy usage. Monitoring and dynamic guidance "
                "could be useful."
            )
        else:
            st.success(
                "Low risk of the lot being at full capacity for this hour."
            )

        # ---------------------------------------------------
        # Predict next 6 hours – use previous predictions as lag features
        # so the curve is not flat and includes synthetic history
        # ---------------------------------------------------
        st.subheader("How does occupancy change over the next 6 hours?")

        hours = []
        occ_vals = []

        # Start from the current prediction as the "latest" occupancy
        prev_occ = occ_pred

        # Initialise lag features (if they exist) with current prediction
        occ_1hr = prev_occ
        occ_24hr = prev_occ
        roll3 = prev_occ

        for h_offset in range(0, 7):
            h_future = (hour + h_offset) % 24

            tmp_input = base_input.copy()

            # update hour
            if "Hour" in tmp_input:
                tmp_input["Hour"] = h_future

            # synthetic history based on previous predictions
            if "Occupancy_1hr_Ago" in tmp_input:
                tmp_input["Occupancy_1hr_Ago"] = occ_1hr
            if "Occupancy_24hr_Ago" in tmp_input:
                tmp_input["Occupancy_24hr_Ago"] = occ_24hr
            if "Occupancy_3hr_Roll_Avg" in tmp_input:
                tmp_input["Occupancy_3hr_Roll_Avg"] = roll3

            x_future = np.array([tmp_input[f] for f in FEATURES]).reshape(1, -1)
            x_future_scaled = scaler.transform(x_future)
            occ_future = best_xgb_reg.predict(x_future_scaled)[0]

            hours.append(h_future)
            occ_vals.append(occ_future)

            # update our "history" for the next step
            occ_24hr = occ_1hr
            occ_1hr = occ_future
            roll3 = (roll3 * 2 + occ_future) / 3.0  # simple rolling average

        fig, ax = plt.subplots()
        ax.plot(hours, occ_vals, marker="o")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Predicted Occupancy")
        ax.set_title(f"Predicted Occupancy Trend – {selected_lot_label or 'Selected Lot'}")
        st.pyplot(fig)

        st.caption(
            "Trend is generated by keeping weather and day constant while changing the hour. "
            "Lag features are updated with previous predictions so the curve reflects "
            "how demand might evolve within this scenario."
        )

# ---------------------------------------------------
# PAGE 3 – LOT STATUS OVERVIEW (ALL LOTS AT ONCE)
# ---------------------------------------------------
if page == "Lot Status Overview":
    st.title("📊 Lot Status Overview – Which Lots Are Likely Full?")

    st.markdown(
        """
        This page shows, for a selected hour and conditions, the predicted:

        - *Occupancy* for each parking lot  
        - *Probability that the lot is near full*  
        - Simple status: 🟥 High risk, 🟧 Busy, 🟩 Comfortable
        """
    )

    # Find lot-related features (one-hot encoded units)
    lot_features = [f for f in FEATURES if f.startswith("Unit_")]
    lot_display_names = [lf.replace("Unit_", "").replace("_", " ") for lf in lot_features]

    if not lot_features:
        st.error(
            "No parking-lot indicator features (starting with 'Unit_') were "
            "found in FEATURES. This view needs those to work."
        )
    else:
        st.subheader("Step 1 – Choose time & weather")

        col1, col2 = st.columns(2)

        with col1:
            month = st.slider("Month (1 = Jan, 12 = Dec)", 1, 12, 7)
            day_of_week = st.slider("Day of Week (0 = Monday, 6 = Sunday)", 0, 6, 5)
            hour = st.slider("Hour of Day", 0, 23, 14)

        with col2:
            max_temp = st.slider("Max Temperature (°C)", -20.0, 40.0, 22.0)
            total_precip = st.slider("Total Precipitation (mm)", 0.0, 30.0, 0.5)
            wind_gust = st.slider("Speed of Max Gust (km/h)", 0.0, 100.0, 12.0)

        is_weekend = 1 if day_of_week in [5, 6] else 0

        st.caption(
            "Lag features (previous-hour occupancy, rolling averages) are set to 0 "
            "for this overview. In a real system they would come from live feeds."
        )

        if st.button("Compute lot status"):
            rows = []

            # Base feature template
            base_input = {f: 0 for f in FEATURES}

            # Common time & weather fields
            if "Month" in base_input:
                base_input["Month"] = month
            if "DayOfWeek" in base_input:
                base_input["DayOfWeek"] = day_of_week
            if "Hour" in base_input:
                base_input["Hour"] = hour
            if "IsWeekend" in base_input:
                base_input["IsWeekend"] = is_weekend
            if "Max Temp (°C)" in base_input:
                base_input["Max Temp (°C)"] = max_temp
            if "Total Precip (mm)" in base_input:
                base_input["Total Precip (mm)"] = total_precip
            if "Spd of Max Gust (km/h)" in base_input:
                base_input["Spd of Max Gust (km/h)"] = wind_gust

            # Loop over each lot, one-hot encode, and predict
            for lot_feat, lot_name in zip(lot_features, lot_display_names):
                lot_input = base_input.copy()
                if lot_feat in lot_input:
                    lot_input[lot_feat] = 1

                x_vec = np.array([lot_input[f] for f in FEATURES]).reshape(1, -1)
                x_scaled = scaler.transform(x_vec)

                occ_pred = best_xgb_reg.predict(x_scaled)[0]
                full_prob = best_xgb_cls.predict_proba(x_scaled)[0, 1]

                if full_prob > 0.7:
                    status = "🟥 High risk full"
                elif full_prob > 0.4:
                    status = "🟧 Busy"
                else:
                    status = "🟩 Comfortable"

                rows.append(
                    {
                        "Lot": lot_name,
                        "Predicted occupancy": occ_pred,
                        "Probability full": full_prob,
                        "Status": status,
                    }
                )

            df = pd.DataFrame(rows)
            df = df.sort_values("Probability full", ascending=False)

            st.subheader("Step 2 – Lot status for selected hour")
            st.dataframe(
                df.style.format(
                    {"Predicted occupancy": "{:.2f}", "Probability full": "{:.1%}"}
                ),
                use_container_width=True,
            )

            st.caption(
                "Lots at the top have the highest probability of being near capacity. "
                "This helps planners direct vehicles to less busy areas."
            )

# ---------------------------------------------------
# PAGE 4 – XAI (EXPLAINABLE AI)
# ---------------------------------------------------
if page == "XAI – Explainable AI":
    st.title("🔍 Explainable AI – Understanding the Models")

    st.markdown(
        """
        This page explains *why* the models make their predictions,
        using Explainable AI tools:

        - *SHAP summary plot*: which features contribute most to predictions  
        - *SHAP bar plot*: overall feature importance  
        - *Partial Dependence Plots (PDPs)*: effect of one feature at a time  
        - *Residual plot*: how close predictions are to the true values  
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
        st.error(f"Could not compute residuals: {e}")
