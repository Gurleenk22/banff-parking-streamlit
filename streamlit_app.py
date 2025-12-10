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

# ---------------------------------------------------
# BASIC PAGE CONFIG + GLOBAL STYLE
# ---------------------------------------------------
st.set_page_config(
    page_title="Banff Parking – ML & XAI Dashboard",
    layout="wide"
)

# Simple modern font + card style
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    }
    .card {
        padding: 1.0rem 1.2rem;
        border-radius: 0.9rem;
        border: 1px solid #e5e7eb;
        background-color: #ffffff;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.06);
    }
    .metric-label {
        font-size: 0.8rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #4b5563;
        margin-top: 0.15rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# OPENAI CLIENT (SAFE – WON'T CRASH IF MISSING)
# ---------------------------------------------------
@st.cache_resource
def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

# ---------------------------------------------------
# LOAD MODELS + DATA (CACHED)
# ---------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """
    Load trained models, scaler, feature list, and test data.

    NOTE: We only use the regression model here.
    "Probability full" is derived from predicted occupancy.
    """
    reg = joblib.load("banff_best_xgb_reg.pkl")      # XGBoost regressor
    scaler = joblib.load("banff_scaler.pkl")         # Scaler used in training
    features = joblib.load("banff_features.pkl")     # List of feature names

    # Test data for XAI and residual analysis
    X_test_scaled = np.load("X_test_scaled.npy")
    y_reg_test = np.load("y_reg_test.npy")

    return reg, scaler, features, X_test_scaled, y_reg_test


best_xgb_reg, scaler, FEATURES, X_test_scaled, y_reg_test = load_models_and_data()

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
            "This is Gurleen's Banff parking assistant. The banff_knowledge.txt "
            "file is missing, so answers are based only on general parking logic "
            "and basic ML explanations."
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
        return (
            "No strong matches in the knowledge base. Answer based on general "
            "parking logic and common ML knowledge."
        )

    return "\n".join(selected)


def generate_chat_answer(user_question, chat_history):
    """
    Calls OpenAI with retrieved context + short chat history.
    If the API is not configured or fails, fall back to a simple
    answer built only from the retrieved context.
    """
    docs, vectorizer, doc_embeddings = load_rag_knowledge()
    context = retrieve_context(user_question, docs, vectorizer, doc_embeddings, k=5)

    client = get_openai_client()
    if client is None:
        # No API key – context-only answer
        return (
            "The OpenAI API key is not configured for this app, so I will answer using "
            "only the project notes.\n\n"
            f"**Most relevant notes:**\n{context}"
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly project assistant helping Gurleen explain a Banff "
                "parking analytics project. Speak clearly and simply to classmates and "
                "instructors who are not data scientists. Use the provided 'Context' as "
                "your main source of truth."
            ),
        },
        {
            "role": "system",
            "content": f"Context from project notes:\n{context}",
        },
    ]

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
        return (
            "I couldn’t contact the language-model service right now. This usually "
            "means the OpenAI API quota or key is not working.\n\n"
            "Here is the most relevant information I can give based only on the "
            "project notes:\n\n"
            f"{context}"
        )

# ---------------------------------------------------
# HELPER: LOT FEATURE NAMES
# ---------------------------------------------------
def get_lot_features():
    lot_features = [f for f in FEATURES if f.startswith("Unit_")]
    lot_display_names = [lf.replace("Unit_", "").replace("_", " ") for lf in lot_features]
    if lot_features:
        lot_pairs = sorted(zip(lot_features, lot_display_names), key=lambda x: x[1])
        lot_features, lot_display_names = zip(*lot_pairs)
        return list(lot_features), list(lot_display_names)
    return [], []


# ---------------------------------------------------
# HELPER: MAP OCCUPANCY -> PROBABILITY + RISK LABEL
# ---------------------------------------------------
def occupancy_to_risk(occ_pred: float):
    """
    Convert occupancy prediction into:
    - approximate probability that lot is near full
    - status label

    Assumes occupancy roughly 0–1. Adjust thresholds if needed.
    """
    # Map occupancy 0.6 -> 0, 1.0 -> 1 (clip to [0,1])
    full_prob = float(np.clip((occ_pred - 0.6) / 0.4, 0.0, 1.0))

    if full_prob > 0.7:
        status = "🟥 High risk full"
    elif full_prob > 0.4:
        status = "🟧 Busy"
    else:
        status = "🟩 Comfortable"

    return full_prob, status


# ---------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------
st.sidebar.title("Banff Parking Dashboard")
st.sidebar.markdown(
    """
    Use this app to:

    - ⚡ Explore hourly parking demand  
    - 📍 Check which lots may be full  
    - 🔍 Understand the model with XAI  
    - 💬 Chat with a parking assistant  
    """
)

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Make Prediction",
        "Lot Status Overview",
        "XAI – Explainable AI",
        "💬 Chat Assistant (RAG)",
    ]
)

# ===================================================
# PAGE 1 – OVERVIEW (CLEAN, SHORT, WITH BOXES)
# ===================================================
if page == "Overview":
    st.title("🚗 Banff Parking Demand – ML Dashboard")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="card">
              <div class="metric-label">Project focus</div>
              <div class="metric-value">Parking demand</div>
              <div class="metric-sub">Tourist season in Banff, May–Sept</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="card">
              <div class="metric-label">Models</div>
              <div class="metric-value">XGBoost</div>
              <div class="metric-sub">Predicts hourly occupancy per lot</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="card">
              <div class="metric-label">Use this for</div>
              <div class="metric-value">Operations</div>
              <div class="metric-sub">Spot high-risk lots & redirect traffic</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    c4, c5 = st.columns(2)
    with c4:
        st.subheader("What the app shows")
        st.markdown(
            """
            - **Single-lot what-if**: pick a lot, hour, and weather → see risk  
            - **All lots overview**: compare which lots are likely full  
            - **Model explainability**: SHAP, PDP and residuals  
            - **Chatbot**: answer questions using your project notes  
            """
        )
    with c5:
        st.subheader("How to demo it quickly")
        st.markdown(
            """
            1. Go to **Make Prediction** → show one busy scenario  
            2. Open **Lot Status Overview** → show all lots at that time  
            3. Visit **XAI** → explain key features (Hour, Month, Temp)  
            4. End with **Chat Assistant** and ask a simple question  
            """
        )

# ===================================================
# PAGE 2 – MAKE PREDICTION
# ===================================================
if page == "Make Prediction":
    st.title("🎯 Single-Lot Prediction – What If?")

    st.caption("Choose a lot, adjust time & weather, then see occupancy and risk.")

    lot_features, lot_display_names = get_lot_features()

    if not lot_features:
        st.warning(
            "No parking-lot indicator features (starting with 'Unit_') were "
            "found in FEATURES. Lot selection is disabled."
        )
    else:
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

        col_lot, col_scen = st.columns([1.2, 1])
        with col_lot:
            selected_lot_label = st.selectbox(
                "Parking lot",
                lot_display_names,
                index=0,
            )
            selected_lot_feature = lot_features[lot_display_names.index(selected_lot_label)]

        with col_scen:
            selected_scenario = st.selectbox(
                "Scenario",
                list(scenario_options.keys()),
                index=1,
            )

        # Default slider values
        default_vals = {"month": 7, "dow": 5, "hour": 13,
                        "max_temp": 22.0, "precip": 0.5, "gust": 12.0}
        if scenario_options[selected_scenario] is not None:
            default_vals.update(scenario_options[selected_scenario])

        st.markdown("### Conditions")

        s1, s2, s3, s4, s5, s6 = st.columns(6)
        with s1:
            month = st.slider("Month", 1, 12, int(default_vals["month"]()_
