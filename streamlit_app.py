import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.inspection import PartialDependenceDisplay

# ==== RAG / Chatbot imports ====
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os

client = OpenAI()  # uses OPENAI_API_KEY from env / Streamlit secrets

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
    """Load trained models, scaler, feature list, and test data."""
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
# SANITY CHECK: MODELS vs FEATURES
# ---------------------------------------------------
def validate_model_feature_alignment():
    """Make sure models, FEATURES list, and scaler all agree on feature count."""
    problems = []

    n_feat = len(FEATURES)

    cls_n = getattr(best_xgb_cls, "n_features_in_", None)
    reg_n = getattr(best_xgb_reg, "n_features_in_", None)

    if cls_n is not None and cls_n != n_feat:
        problems.append(
            f"• Classifier expects {cls_n} features but FEATURES has {n_feat}."
        )

    if reg_n is not None and reg_n != n_feat:
        problems.append(
            f"• Regressor expects {reg_n} features but FEATURES has {n_feat}."
        )

    # Try a dummy transform to check scaler shape
    try:
        _dummy = np.zeros((1, n_feat))
        _ = scaler.transform(_dummy)
    except Exception as e:
        problems.append(f"• Scaler cannot transform a vector of length {n_feat}: {e}")

    if problems:
        st.error(
            "❌ Model / feature mismatch detected.\n\n"
            + "\n".join(problems)
            + "\n\n**Fix:** Open your training notebook and re-save, from the SAME run:\n"
            "- `banff_best_xgb_reg.pkl`\n"
            "- `banff_best_xgb_cls.pkl`\n"
            "- `banff_scaler.pkl`\n"
            "- `banff_features.pkl`\n"
            "- `X_test_scaled.npy`\n"
            "- `y_reg_test.npy`"
        )
        st.stop()

validate_model_feature_alignment()

# ---------------------------------------------------
# RAG: LOAD KNOWLEDGE + BUILD VECTORIZER
# ---------------------------------------------------
@st.cache_resource
def load_rag_knowledge():
    """
    Loads banff_knowledge.txt and builds TF-IDF vectors.
    Each non-empty line is treated as a small document.
    """
    knowledge_path = "banff_knowledge.txt"

    if not os.path.exists(knowledge_path):
        docs = [
            "This is Gurleen's Banff parking assistant. The banff_knowledge.txt file is "
            "missing, so answers are based only on general parking logic."
        ]
    else:
        with open(knowledge_path, "r", encoding="utf-8") as f:
            docs = [line.strip() for line in f.readlines() if line.strip()]

    vectorizer = TfidfVectorizer(stop_words="english")
    doc_embeddings = vectorizer.fit_transform(docs)

    return docs, vectorizer, doc_embeddings


def retrieve_context(query, docs, vectorizer, doc_embeddings, k=5):
    """Returns top-k most relevant lines from the knowledge base."""
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, doc_embeddings).flatten()
    top_idx = sims.argsort()[::-1][:k]
    selected = [docs[i] for i in top_idx if sims[i] > 0.0]

    if not selected:
        return "No strong matches in the knowledge base. Answer based on general parking logic."

    return "\n".join(selected)


def generate_chat_answer(user_question, chat_history):
    """
    Calls OpenAI with retrieved context + short chat history.
    If the API fails (e.g., insufficient_quota), fall back to
    a simple answer built only from the retrieved context.
    """
    docs, vectorizer, doc_embeddings = load_rag_knowledge()
    context = retrieve_context(user_question, docs, vectorizer, doc_embeddings, k=5)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly project assistant helping Gurleen explain a Banff "
                "parking analytics project. Speak clearly and simply, as if you are "
                "presenting to classmates and instructors who are not data scientists. "
                "Use the provided 'Context' from the project notes as your main source "
                "of truth. If the context does not clearly contain the answer, say that "
                "openly and give a short, reasonable guess based on typical parking "
                "behaviour."
            ),
        },
        {
            "role": "system",
            "content": f"Context from project notes:\n{context}",
        },
    ]

    # keep last few turns of history
    for h in chat_history[-4:]:
        messages.append(
            {
                "role": h["role"],
                "content": h["content"],
            }
        )

    messages.append({"role": "user", "content": user_question})

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        # Friendly fallback when quota is exhausted or API not reachable
        return (
            "I couldn’t contact the language-model service right now "
            "(this usually means the OpenAI API quota or free credits are used up "
            "for this key).\n\n"
            "Here is the most relevant information I can give based only on "
            "the project notes:\n\n"
            f"{context}"
        )

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
    - Chat with a **parking assistant** using RAG  
    """
)

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "App Guide – What This Dashboard Does",
        "Make Prediction",
        "Lot Status Overview",
        "XAI – Explainable AI",
        "💬 Chat Assistant (RAG)",
    ]
)

# ---------------------------------------------------
# PAGE 1 – OVERVIEW
# ---------------------------------------------------
if page == "Overview":
    st.title("🚗 Banff Parking Demand – Machine Learning Overview")

    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.markdown(
            """
            ### Project Question

            **How can Banff use real data to anticipate parking pressure and avoid full lots during the May–September tourist season?**

            This project combines:
  
            - **Parking management data** – when and where people park  
            - **Weather data** – temperature, rain, and wind  
            - **Engineered features** – hour, weekday/weekend, lagged occupancy, rolling averages  

            A Gradient-boosted tree model (**XGBoost**) predicts:
            - Hourly **occupancy level** for each lot  
            - **Probability that a lot is near full** (> 90% capacity)  
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
            ✅ Models trained on **historical hourly data**  
            ✅ Includes **time, weather, and history** features  
            ✅ Deployed as this **Streamlit decision-support app**
            """
        )

    st.markdown("---")

    st.subheader("How to Use This App")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            **1. Make Prediction**  
            - Choose a **lot & scenario**  
            - Adjust **time & weather**  
            - See predicted occupancy & full-lot risk
            """
        )

    with col2:
        st.markdown(
            """
            **2. Lot Status Overview**  
            - Select a **single hour**  
            - Compare **all lots**  
            - Status: 🟥 High risk full, 🟧 Busy, 🟩 Comfortable  
            - Supports operational decisions & signage
            """
        )

    with col3:
        st.markdown(
            """
            **3. XAI – Explainable AI**  
            - Global **SHAP** feature importance  
            - **Partial Dependence Plots** (Hour, Month, Temp)  
            - **Residual plot** to check model fit  
            - Helps justify decisions to stakeholders
            """
        )

    st.info(
        "Tip: move between pages using the left sidebar. Start with "
        "**App Guide** if you want an explanation of all pages; then try "
        "**Make Prediction** to see how the model behaves for different scenarios."
    )

# ---------------------------------------------------
# PAGE 2 – APP GUIDE (NEW PAGE)
# ---------------------------------------------------
if page == "App Guide – What This Dashboard Does":
    st.title("📘 App Guide – What Each Page Shows")

    st.markdown(
        """
        This page is like a tour guide for your dashboard.  
        It explains, in simple language, what happens on the other pages and
        how an operator or instructor should use them.
        """
    )

    st.markdown("### 🎯 Big Picture – What problem are we solving?")
    st.markdown(
        """
        Banff gets very busy in the tourist season. When parking lots suddenly fill up,
        visitors get frustrated and traffic becomes messy.

        This dashboard uses **machine learning** to:
        - Predict **how full each lot will be** at a specific hour  
        - Estimate the **risk that a lot is near full**  
        - Help staff **redirect visitors** to quieter lots  
        - Explain *why* the model thinks a lot will be busy (XAI)
        """
    )

    st.markdown("---")

    # --- Make Prediction explanation ---
    st.subheader("1️⃣ Make Prediction – “What if” for one parking lot")
    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown(
            """
            On this page you simulate one specific parking lot.

            **Inputs on the left:**
            - Select a **parking lot** from the list (e.g., *BANFF02 – WOLF AT MARTEN*)  
            - Pick a **scenario** like *Sunny Weekend Midday* or *Rainy Weekday Afternoon*  
            - Adjust sliders for:
              - Month, day of week, hour  
              - Temperature, rain, and wind  

            **Outputs on the right:**
            - **Predicted occupancy (model units)** – higher value = more cars  
            - **Probability the lot is near full** – shown as a percentage  
            - Colour message:
              - 🟥 High risk full  
              - 🟧 Moderate / busy  
              - 🟩 Low risk  

            You can talk about this page as:  
            *“Here the operations team can test different ‘what-if’ cases for a single lot
            before a busy day, so they know where pressure will build up first.”*
            """
        )

    with col2:
        st.markdown("**Good questions to explore on this page:**")
        st.markdown(
            """
            - *What happens to BANFF02 on a sunny Saturday at 2pm?*  
            - *How does the risk change if the weather is cold and rainy?*  
            - *Which lot stays comfortable longer in the evening?*
            """
        )

    st.markdown("---")

    # --- Lot Status Overview explanation ---
    st.subheader("2️⃣ Lot Status Overview – Compare all lots at once")
    col3, col4 = st.columns([1.4, 1])

    with col3:
        st.markdown(
            """
            This page answers: **“At this hour, which lots are in trouble?”**

            **Inputs:**
            - One set of sliders for **time and weather** (month, day, hour, temp, rain, wind)  

            **Outputs:**
            - A table where **each row is a parking lot**  
            - For every lot you see:
              - Predicted occupancy  
              - Probability the lot is full  
              - Status with colour:
                - 🟥 High risk full (row tinted light red)  
                - 🟧 Busy (row tinted light orange)  
                - 🟩 Comfortable (row tinted light green)  

            Lots are shown in **numeric order**, so BANFF02, BANFF03, BANFF04, etc.,
            are easy to read as a group.
            """
        )

    with col4:
        st.markdown("**How staff could use this page:**")
        st.markdown(
            """
            - Quickly check the **next hour** before a shift starts  
            - Decide where to place **signs or staff** to redirect cars  
            - Spot **which lots usually hit high risk first** during busy days
            """
        )

    st.markdown("---")

    # --- XAI explanation ---
    st.subheader("3️⃣ XAI – Explainable AI – Why the model makes these predictions")
    col5, col6 = st.columns([1.4, 1])

    with col5:
        st.markdown(
            """
            This page is for explaining the **logic behind the model** to instructors,
            stakeholders, or anyone who asks *“Why should we trust this?”*  

            It includes:

            - **SHAP Summary Plot**  
              Shows which features (Hour, Month, Weather, etc.) push predictions
              up or down. Each dot is one observation.

            - **SHAP Bar Plot (Feature Importance)**  
              Ranks features by how much they influence occupancy overall.

            - **Partial Dependence Plots (PDPs)**  
              Show the *average effect* of one feature at a time
              (e.g., how occupancy changes through the day, or by temperature).

            - **Residual Plot**  
              Compares predicted vs actual values. If the points are spread around
              the zero line, the model is not heavily biased.
            """
        )

    with col6:
        st.markdown("**Nice talking points here:**")
        st.markdown(
            """
            - *“Hour of the day and month of the year are the strongest drivers.”*  
            - *“Weather has an effect – on cold or rainy days occupancy is different.”*  
            - *“Residuals show the model is generally accurate without big bias.”*
            """
        )

    st.markdown("---")

    # --- Chat assistant explanation ---
    st.subheader("4️⃣ Chat Assistant (RAG) – Ask questions in plain English")
    col7, col8 = st.columns([1.4, 1])

    with col7:
        st.markdown(
            """
            This page turns the project notes into a **question-answer helper**.

            Behind the scenes it:
            1. Reads lines from `banff_knowledge.txt` (your project notes).  
            2. Finds the most relevant lines for the user’s question.  
            3. Uses an OpenAI model to write a friendly answer, grounded in those notes.  

            This is useful when someone asks:
            - *“Which lots usually get full first?”*  
            - *“What variables did you include in the model?”*  
            - *“How did you clean the data?”*
            """
        )

    with col8:
        st.markdown("**Example questions you can type live in class:**")
        st.markdown(
            """
            - *“Explain in simple words how this model predicts parking demand.”*  
            - *“Why did we choose XGBoost instead of a simple linear model?”*  
            - *“How could Banff staff actually use these predictions day-to-day?”*
            """
        )

    st.success(
        "During your presentation you can start on this page, give a quick tour of "
        "each part of the dashboard, and then jump into a live demo on the other pages."
    )

# ---------------------------------------------------
# PAGE 3 – MAKE PREDICTION (NO FUTURE GRAPH)
# ---------------------------------------------------
if page == "Make Prediction":
    st.title("🎯 Interactive Parking Demand Prediction")

    st.markdown(
        """
        Use this page to explore *what-if* scenarios for a single Banff parking lot.

        1. Select a **parking lot**  
        2. Choose a **scenario** (or adjust the sliders)  
        3. See:
           - Predicted **occupancy** for the selected hour  
           - **Probability** the lot is near full  
        """
    )

    # Find lot indicator features (one-hot encoded units)
    lot_features = [f for f in FEATURES if f.startswith("Unit_")]
    lot_display_names = [lf.replace("Unit_", "").replace("_", " ") for lf in lot_features]

    # Sort lot list alphabetically so numbers appear in order (BANFF02, BANFF03, …)
    if lot_features:
        lot_pairs = sorted(zip(lot_features, lot_display_names), key=lambda x: x[1])
        lot_features, lot_display_names = zip(*lot_pairs)
        lot_features = list(lot_features)
        lot_display_names = list(lot_display_names)

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
        "Lag features (previous-hour occupancy, rolling averages) are set automatically "
        "by the model and are not entered manually here."
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
        try:
            # Current-hour predictions
            occ_pred = float(best_xgb_reg.predict(x_scaled)[0])
            full_prob = float(best_xgb_cls.predict_proba(x_scaled)[0, 1])
        except ValueError as e:
            st.error(
                "❌ There is a mismatch between the input features and the classifier.\n\n"
                f"Technical details: `{e}`\n\n"
                "This usually means `banff_best_xgb_cls.pkl` was trained with a different "
                "set of features than `banff_features.pkl` / `banff_scaler.pkl`.\n\n"
                "Please re-export those files together from the same notebook run."
            )
            st.stop()

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
                "⚠️ High risk this lot will be full. Consider redirecting drivers "
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
# PAGE 4 – LOT STATUS OVERVIEW (ALL LOTS AT ONCE)
# ---------------------------------------------------
if page == "Lot Status Overview":
    st.title("📊 Lot Status Overview – Which Lots Are Likely Full?")

    st.markdown(
        """
        This page shows, for a selected hour and conditions, the predicted:

        - **Occupancy** for each parking lot  
        - **Probability that the lot is near full**  
        - Simple status: 🟥 High risk, 🟧 Busy, 🟩 Comfortable
        """
    )

    lot_features = [f for f in FEATURES if f.startswith("Unit_")]
    lot_display_names = [lf.replace("Unit_", "").replace("_", " ") for lf in lot_features]

    # sort lots alphabetically so numbers are in sequence
    if lot_features:
        lot_pairs = sorted(zip(lot_features, lot_display_names), key=lambda x: x[1])
        lot_features, lot_display_names = zip(*lot_pairs)
        lot_features = list(lot_features)
        lot_display_names = list(lot_display_names)

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
