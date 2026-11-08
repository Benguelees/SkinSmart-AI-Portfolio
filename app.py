import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
from rag.engine import get_query_engine

from recommender import (
    get_app_recommendations,
    recommend_products_v2,
    recommend_products_hybrid
)

@st.cache_resource
def load_rag_engine():
    """Load the SkinSmart RAG engine once and reuse it."""
    return get_query_engine()

# --------------------------------------------------
# Page setup
# --------------------------------------------------

st.set_page_config(
    page_title="SkinSmart AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# Navigation helper
# --------------------------------------------------

def go_to_skin_profile():
    st.session_state["page"] = "Skin Profile"


# --------------------------------------------------
# Navigation
# --------------------------------------------------

page = st.sidebar.radio(
    "Navigate",
    [
    "Home",
    "Skin Profile",
    "Recommendations",
    "Ask SkinSmart",
    "Behind the Model"
    ],
    key="page"
)


# --------------------------------------------------
# Custom styling
# --------------------------------------------------

st.markdown(
    """
<style>

/* ==================================================
   MAIN APP
================================================== */

.stApp {
    background: linear-gradient(
        180deg,
        #f8f4f1 0%,
        #f3eee9 100%
    );
}

.main-title {
    font-size: 3.4rem;
    font-weight: 800;
    color: #2f2b2b !important;
    margin-bottom: 0.3rem;
}

.subtitle {
    font-size: 1.15rem;
    color: #766f6b !important;
    margin-bottom: 2.2rem;
}

.disclaimer {
    font-size: 0.82rem;
    color: #857c77 !important;
    margin-top: 2rem;
}


/* ==================================================
   MAIN CONTENT TEXT
================================================== */

[data-testid="stAppViewContainer"] .main {
    color: #332d2b !important;
}

[data-testid="stAppViewContainer"] .main p,
[data-testid="stAppViewContainer"] .main label {
    color: #332d2b;
}


/* ==================================================
   HEADINGS
================================================== */

[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stHeading"] h1,
[data-testid="stHeading"] h2,
[data-testid="stHeading"] h3,
[data-testid="stHeading"] h4,
.main h1,
.main h2,
.main h3,
.main h4 {
    color: #332d2b !important;
}


/* ==================================================
   SIDEBAR
================================================== */

section[data-testid="stSidebar"] {
    background-color: #272832 !important;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #f5f0ed !important;
}

section[data-testid="stSidebar"]
[role="radiogroup"] label p {
    color: #f5f0ed !important;
}

section[data-testid="stSidebar"]
[data-testid="stMarkdownContainer"] p {
    color: #f5f0ed !important;
}


/* ==================================================
   FORM LABELS
================================================== */

[data-testid="stWidgetLabel"] p {
    color: #332d2b !important;
}

.stCheckbox label p {
    color: #332d2b !important;
}

.stSelectbox label p,
.stMultiSelect label p {
    color: #332d2b !important;
}


/* ==================================================
   SELECT BOXES
================================================== */

[data-baseweb="select"] > div {
    background-color: #fbf8f6 !important;
    color: #332d2b !important;
    border-color: #d8c7bf !important;
    box-shadow: none !important;
}

[data-baseweb="select"] input {
    color: #332d2b !important;
    background-color: transparent !important;
}

[data-baseweb="select"] input::placeholder {
    color: #776b66 !important;
    opacity: 1 !important;
}

[data-baseweb="select"] span {
    color: #332d2b !important;
}

[data-baseweb="select"] svg {
    fill: #6b5a53 !important;
    color: #6b5a53 !important;
}

[data-baseweb="select"]:focus-within > div {
    border-color: #a7776a !important;
    box-shadow: 0 0 0 1px #a7776a !important;
}


/* ==================================================
   DROPDOWN MENU
================================================== */

[data-baseweb="popover"] {
    background-color: #fbf8f6 !important;
}

[data-baseweb="popover"] ul,
[data-baseweb="popover"] [role="listbox"] {
    background-color: #fbf8f6 !important;
}

[data-baseweb="popover"] [role="option"] {
    background-color: #fbf8f6 !important;
    color: #332d2b !important;
}

[data-baseweb="popover"] [role="option"] * {
    color: #332d2b !important;
}

[data-baseweb="popover"] [role="option"]:hover {
    background-color: #efe4de !important;
}


/* ==================================================
   MULTISELECT TAGS
================================================== */

[data-baseweb="tag"] {
    background-color: #d8b4a6 !important;
}

[data-baseweb="tag"] span,
[data-baseweb="tag"] svg {
    color: #4b342d !important;
}


/* ==================================================
   CHECKBOXES
================================================== */

[data-testid="stCheckbox"] label span {
    color: #332d2b !important;
}


/* ==================================================
   BUTTONS
================================================== */

div.stButton > button {
    background: #a7776a;
    color: #ffffff !important;
    border: none;
    border-radius: 14px;
    font-weight: 700;
}

div.stButton > button p {
    color: #ffffff !important;
}

div.stButton > button:hover {
    background: #8f6258;
    color: #ffffff !important;
    border: none;
}


/* ==================================================
   CARDS
================================================== */

[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.72);
    border-color: rgba(90, 75, 70, 0.14) !important;
    border-radius: 20px;
    box-shadow: 0 6px 24px rgba(60, 45, 40, 0.05);
}

[data-testid="stVerticalBlockBorderWrapper"] h1,
[data-testid="stVerticalBlockBorderWrapper"] h2,
[data-testid="stVerticalBlockBorderWrapper"] h3,
[data-testid="stVerticalBlockBorderWrapper"] h4 {
    color: #332d2b !important;
}


/* ==================================================
   PROGRESS BAR
================================================== */

[data-testid="stProgress"] > div > div > div > div {
    background-color: #a7776a !important;
}

[data-testid="stProgress"] > div > div {
    background-color: #eadfd9 !important;
}


/* ==================================================
   METRICS
================================================== */

[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.42);
    padding: 12px 16px;
    border-radius: 14px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #6f6661 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #332d2b !important;
    font-weight: 700 !important;
}


/* ==================================================
   CAPTIONS
================================================== */

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #7b736e !important;
}
/* Force regular main-page text to stay dark */

[data-testid="stAppViewContainer"] .main strong,
[data-testid="stAppViewContainer"] .main b,
[data-testid="stAppViewContainer"] .main p,
[data-testid="stAppViewContainer"] .main div:not([data-testid="stSidebar"]) {
    color: #332d2b;
}

/* Keep buttons white */
div.stButton > button,
div.stButton > button p {
    color: #ffffff !important;
}

/* Captions stay muted */
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #7b736e !important;
}

/* Metric text */
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #6f6661 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #332d2b !important;
}

/* Sidebar stays light */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #f5f0ed !important;
}

[data-testid="stChatMessage"] {
    color: #2f2a28;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] h4,
[data-testid="stChatMessage"] strong {
    color: #2f2a28 !important;
}

[data-testid="stChatMessageContent"] {
    color: #2f2a28 !important;
}

</style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Load prepared data
# --------------------------------------------------

data_path = Path("data/skincare_product_profiles.csv")
profiles = pd.read_csv(data_path)


# --------------------------------------------------
# Shared header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">SkinSmart AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="subtitle">
Personalized skincare recommendations based on your skin profile
and ingredient preferences.
</div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# HOME
# ==================================================

if page == "Home":

    hero_left, hero_right = st.columns([1.3, 0.7])

    with hero_left:

        st.markdown(
            """
<div style="
display:inline-block;
background:#efe4de;
color:#6b4f46;
padding:7px 14px;
border-radius:999px;
font-weight:700;
margin-bottom:18px;
">
✨ Personalized skincare intelligence
</div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
<h1 style="
font-size:3rem;
line-height:1.08;
margin-bottom:18px;
color:#332d2b;
">
Find skincare that fits<br>
your skin — not everyone else's.
</h1>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
<p style="
color:#332d2b;
font-size:1rem;
line-height:1.65;
margin-bottom:10px;
">
SkinSmart AI combines explainable profile matching with machine-learning
feature similarity to rank skincare products around your individual skin needs.
</p>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
<p style="
color:#6f625d;
font-size:0.94rem;
line-height:1.55;
margin-top:4px;
margin-bottom:18px;
">
Designed as a transparent recommendation layer for skincare discovery,
personalization and e-commerce experiences.
</p>
            """,
            unsafe_allow_html=True
        )

        st.button(
            "Start your skin analysis →",
            type="primary",
            width="content",
            on_click=go_to_skin_profile
        )

    with hero_right:

        st.markdown(
            """<div style="
background:linear-gradient(145deg,#ead8cf,#f6eeea);
padding:32px;
border-radius:28px;
min-height:250px;
box-shadow:0 10px 35px rgba(80,60,50,0.08);
">

<div style="
font-size:1rem;
color:#806c64;
margin-bottom:18px;
">
SKINSMART PROFILE
</div>

<div style="
font-size:1.7rem;
font-weight:750;
color:#332d2b;
margin-bottom:22px;
">
Your skin becomes the starting point.
</div>

<div style="
background:rgba(255,255,255,0.68);
color:#332d2b;
padding:12px 16px;
border-radius:14px;
margin-bottom:10px;
">
💧 Skin type + hydration needs
</div>

<div style="
background:rgba(255,255,255,0.68);
color:#332d2b;
padding:12px 16px;
border-radius:14px;
margin-bottom:10px;
">
🌿 Sensitivity + ingredient preferences
</div>

<div style="
background:rgba(255,255,255,0.68);
color:#332d2b;
padding:12px 16px;
border-radius:14px;
">
✨ Personalized product ranking
</div>

</div>""",
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown("### Built around your skin")

    benefit1, benefit2, benefit3 = st.columns(3)

    with benefit1:
        with st.container(border=True):
            st.markdown("### 💧 Profile-aware")
            st.write(
                "Recommendations adapt to skin type, sensitivity, "
                "acne tendency and skincare goals."
            )

    with benefit2:
        with st.container(border=True):
            st.markdown("### 🧪 Ingredient-aware")
            st.write(
                "Products are evaluated using ingredient-derived features "
                "such as hydration, barrier support and irritancy."
            )

    with benefit3:
        with st.container(border=True):
            st.markdown("### ✨ Hybrid AI recommendation")
            st.write(
                "Explainable profile scoring is combined with machine-learning "
                "feature similarity while keeping the ranking transparent."
            )

    st.divider()

    st.markdown("### From skin profile to personalized matches")

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown("#### 01 · Tell SkinSmart about your skin")
        st.caption(
            "Choose your skin type, sensitivity, acne tendency "
            "and skincare goals."
        )

    with step2:
        st.markdown("#### 02 · Analyze with a hybrid engine")
        st.caption(
            "Explainable skincare features are combined with "
            "machine-learning nearest-neighbor similarity."
        )

    with step3:
        st.markdown("#### 03 · Rank your best matches")
        st.caption(
            "Hard preferences are applied before the two signals are "
            "combined into the final personalized ranking."
        )

    st.divider()

    st.markdown("### The data behind SkinSmart")

    stat1, stat2, stat3 = st.columns(3)

    with stat1:
        st.metric(
            "Products analyzed",
            len(profiles)
        )

    with stat2:
        st.metric(
            "Engineered data features",
            len(profiles.columns)
        )

    with stat3:
        st.metric(
            "Recommendation approach",
            "Hybrid AI"
        )

    st.caption(
        "Product and ingredient information is transformed into structured "
        "features that power an explainable + machine-learning recommendation engine."
    )

    st.caption(
        "Built with Python · pandas · scikit-learn · Streamlit"
    )

    st.write("")

    st.button(
        "Build my skin profile",
        use_container_width=True,
        on_click=go_to_skin_profile
    )


# ==================================================
# SKIN PROFILE
# ==================================================

elif page == "Skin Profile":

    st.header("Tell SkinSmart what your skin needs")

    st.write(
        "Build a simple profile from your skin characteristics, goals and "
        "ingredient preferences. SkinSmart uses these signals to personalize "
        "the final ranking."
    )

    label1, label2 = st.columns(2)

    with label1:
        st.markdown("### Skin characteristics")

    with label2:
        st.markdown("### Goals & preferences")

    col1, col2 = st.columns(2)

    with col1:

        skin_type = st.selectbox(
            "Skin type",
            ["normal", "dry", "oily"],
            format_func=lambda x: x.title()
        )

        sensitive = st.checkbox(
            "Sensitive skin"
        )

        acne_prone = st.checkbox(
            "Acne-prone skin"
        )

    with col2:

        concerns = st.multiselect(
            "What are your main skincare goals?",
            [
                "hydration",
                "barrier_support",
                "antioxidant_support"
            ],
            format_func=lambda x: {
                "hydration": "Hydration",
                "barrier_support": "Skin barrier support",
                "antioxidant_support": "Antioxidant support"
            }[x]
        )

        fragrance_free = st.checkbox(
            "Prefer fragrance-free products"
        )

        avoid_drying_alcohol = st.checkbox(
            "Avoid drying alcohol"
        )

    max_comedogenicity = st.selectbox(
        "Maximum pore-clogging tolerance",
        [None, 0, 1, 2, 3, 4, 5],
        format_func=lambda x: (
            "No limit"
            if x is None
            else str(x)
        )
    )

    if st.button(
        "Create my personalized matches",
        type="primary",
        use_container_width=True
    ):

        st.session_state["user_profile"] = {
            "skin_type": skin_type,
            "sensitive": sensitive,
            "acne_prone": acne_prone,
            "fragrance_free": fragrance_free,
            "avoid_drying_alcohol": avoid_drying_alcohol,
            "max_comedogenicity": max_comedogenicity,
            "concerns": concerns
        }

        if "recommendations" in st.session_state:
            del st.session_state["recommendations"]

        st.success(
            "Your SkinSmart profile is ready. "
            "Open Recommendations to generate your personalized matches."
        )


# ==================================================
# RECOMMENDATIONS
# ==================================================

elif page == "Recommendations":

    st.header("Your personalized matches")

    if "user_profile" not in st.session_state:
        st.info(
            "Please create your skin profile first using the Skin Profile page."
        )

    else:
        user_profile = st.session_state["user_profile"]

        st.markdown("### Your current profile")

        profile1, profile2, profile3 = st.columns(3)

        with profile1:
            st.write(f"**Skin type:** {user_profile['skin_type'].title()}")

        with profile2:
            st.write(
                f"**Sensitive:** "
                f"{'Yes' if user_profile['sensitive'] else 'No'}"
            )

        with profile3:
            st.write(
                f"**Acne-prone:** "
                f"{'Yes' if user_profile['acne_prone'] else 'No'}"
            )

        concern_labels = {
            "hydration": "Hydration",
            "barrier_support": "Skin barrier support",
            "antioxidant_support": "Antioxidant support"
        }

        selected_goals = [
            concern_labels.get(concern, concern.replace("_", " ").title())
            for concern in user_profile.get("concerns", [])
        ]

        selected_preferences = []

        if user_profile["fragrance_free"]:
            selected_preferences.append("Fragrance-free")

        if user_profile["avoid_drying_alcohol"]:
            selected_preferences.append("No drying alcohol")

        if user_profile["max_comedogenicity"] is not None:
            selected_preferences.append(
                "Pore-clogging tolerance ≤ "
                f"{user_profile['max_comedogenicity']}"
            )

        st.caption(
            f"Goals: {', '.join(selected_goals) if selected_goals else 'None selected'}"
            "  •  "
            f"Preferences: "
            f"{', '.join(selected_preferences) if selected_preferences else 'No additional filters'}"
        )

        has_personalization_signal = (
            user_profile["skin_type"] != "normal"
            or user_profile["sensitive"]
            or user_profile["acne_prone"]
            or bool(user_profile["concerns"])
        )

        recommendation_button_label = (
            "Refresh my SkinSmart matches"
            if "recommendations" in st.session_state
            else "Find my SkinSmart matches"
        )

        if st.button(
            recommendation_button_label,
            type="primary",
            use_container_width=True
        ):
            if not has_personalization_signal:
                if "recommendations" in st.session_state:
                    del st.session_state["recommendations"]

                st.warning(
                    "Your current profile does not contain enough "
                    "personalization signals yet. Select at least one "
                    "skincare goal, sensitivity option, acne-prone skin, "
                    "or choose dry/oily skin to generate meaningful matches."
                )
            else:
                recommendations = get_app_recommendations(
                    profiles,
                    user_profile,
                    top_n=5
                )
                st.session_state["recommendations"] = recommendations

                st.session_state["recommendations"] = recommendations

                

        if "recommendations" in st.session_state:
            recommendations = st.session_state["recommendations"]

            st.divider()

            st.caption(
                "Your Top 5 are ranked by SkinSmart's hybrid engine. "
                "Open Recommendation intelligence to inspect the scoring details."
            )

            if recommendations.empty:
                st.warning(
                    "No products matched all of your selected preferences. "
                    "Try relaxing one of your filters."
                )
            else:
                max_score = recommendations["personalized_score"].max()

                for rank, (_, row) in enumerate(
                    recommendations.iterrows(),
                    start=1
                ):
                    if rank == 1:
                        st.markdown(
                            """<div style="
background:#d8b4a6;
color:#4b342d;
display:inline-block;
padding:6px 12px;
border-radius:999px;
font-weight:700;
margin-bottom:10px;
">
✨ Top Match
</div>""",
                            unsafe_allow_html=True
                        )

                    with st.container(border=True):
                        st.markdown(
                             f"<p style='color: #655751 !important; font-size: 0.85rem;'>"
                             f"MATCH #{rank} · {row['brand'].upper()}"
                             "</p>",
                             unsafe_allow_html=True
                        )

                        st.subheader(row["product_name"])

                        if max_score > 0:
                            relative_match = (
                                row["personalized_score"] / max_score
                            )
                        else:
                            relative_match = 0.0

                        relative_match = max(
                            0.0,
                            min(float(relative_match), 1.0)
                        )

                        st.markdown(
                             f'<p style="color: #655751 !important; font-size: 0.9rem;">'
                             f'Relative strength within your personalized Top 5: {relative_match:.0%}'
                             f'</p>',
                             unsafe_allow_html=True
                        )

                        st.progress(relative_match)

                        def signal_label(value):
                            if value >= 0.70:
                                return "Very strong"
                            elif value >= 0.45:
                                return "Strong"
                            elif value >= 0.20:
                                return "Moderate"
                            elif value > 0:
                                return "Low"
                            return "Minimal"

                        with st.expander(
                            "Recommendation intelligence",
                            expanded=(rank == 1)
                        ):
                            hybrid1, hybrid2, hybrid3 = st.columns(3)

                            with hybrid1:
                                st.metric(
                                    "Final Hybrid Match",
                                    f"{row['hybrid_score']:.2f}"
                                )

                            with hybrid2:
                                st.metric(
                                    "Profile-fit signal",
                                    signal_label(
                                        row["v2_weighted_contribution"]
                                    )
                                )
                                st.caption(
                                    "70% of the final ranking"
                                )

                            with hybrid3:
                                st.metric(
                                    "ML feature similarity",
                                    signal_label(
                                        row["ml_weighted_contribution"]
                                    )
                                )
                                st.caption(
                                    "30% of the final ranking"
                                )

                            st.caption(
                                "SkinSmart combines two independent signals. "
                                "A product can rank highly because it is an exceptionally "
                                "strong profile fit even when its machine-learning "
                                "feature similarity is lower."
                            )

                            st.markdown("### Profile factors")

                            score_parts = []

                            if row["skin_type_contribution"] > 0:
                                score_parts.append(
                                    ("Skin type fit", row["skin_type_contribution"])
                                )

                            if row["sensitivity_contribution"] > 0:
                                score_parts.append(
                                    ("Sensitivity fit", row["sensitivity_contribution"])
                                )

                            if row["acne_contribution"] > 0:
                                score_parts.append(
                                    ("Acne-prone fit", row["acne_contribution"])
                                )

                            if row["hydration_contribution"] > 0:
                                score_parts.append(
                                    ("Hydration goal", row["hydration_contribution"])
                                )

                            if row["barrier_contribution"] > 0:
                                score_parts.append(
                                    ("Barrier support", row["barrier_contribution"])
                                )

                            if row["antioxidant_contribution"] > 0:
                                score_parts.append(
                                    (
                                        "Antioxidant support",
                                        row["antioxidant_contribution"]
                                    )
                                )

                            if score_parts:
                                score_cols = st.columns(
                                    min(len(score_parts), 4)
                                )

                                for i, (label, value) in enumerate(score_parts):
                                    with score_cols[i % len(score_cols)]:
                                        st.metric(label, f"{value:.2f}")

                            st.markdown("**Ingredient evidence**")

                            factor1, factor2, factor3 = st.columns(3)

                            with factor1:
                                st.metric(
                                    "Hydration ingredients",
                                    int(row["humectant_count"])
                                )

                            with factor2:
                                st.metric(
                                    "Barrier-support ingredients",
                                    int(row["skin_protecting_count"])
                                )

                            with factor3:
                                st.metric(
                                    "Emollient ingredients",
                                    int(row["emollient_count"])
                                )

                            st.markdown(
                                "**Why this matches your profile**"
                            )

                            reasons = [
                                reason.strip()
                                for reason in row["why_recommended"].split(";")
                                if reason.strip()
                            ]

                            reasons = list(dict.fromkeys(reasons))

                            if reasons:
                                badge_cols = st.columns(
                                    min(len(reasons), 4)
                                )

                                for i, reason in enumerate(reasons):
                                    with badge_cols[i % len(badge_cols)]:
                                        st.markdown(
                                            f"""<div style="
    background:#efe4de;
    color:#5c4840;
    padding:8px 12px;
    border-radius:999px;
    text-align:center;
    margin-bottom:8px;
    font-size:0.9rem;
    font-weight:600;
    ">
    {reason.title()}
    </div>""",
                                            unsafe_allow_html=True
                                        )

                        st.write("")


    # ==================================================
    # BEHIND THE MODEL
    # ==================================================
elif page == "Behind the Model":

    st.header(
        "Behind the recommendation engine"
    )

    st.write(
        "SkinSmart AI uses a hybrid recommendation architecture: 70% "
        "explainable profile-based scoring and 30% machine-learning "
        "nearest-neighbor similarity."
    )

    with st.expander(
        "Model card",
        expanded=False
    ):
        model1, model2, model3 = st.columns(3)

        with model1:
            st.markdown("**Architecture**")
            st.caption("Hybrid recommender")

        with model2:
            st.markdown("**ML method**")
            st.caption("StandardScaler + Nearest Neighbors")

        with model3:
            st.markdown("**Validation**")
            st.caption("6 profiles · 20% median Top-5 overlap · 0 filter violations")

        st.caption(
            "The ML layer is unsupervised and measures feature-space similarity. "
            "The explainable layer keeps the ranking interpretable."
        )

    st.markdown(
        "### Dataset overview"
    )

    data1, data2, data3 = st.columns(3)

    with data1:

        st.metric(
            "Products analyzed",
            len(profiles)
        )

    with data2:

        st.metric(
            "Engineered data features",
            len(profiles.columns)
        )

    with data3:

        st.metric(
            "ML feature inputs",
            9
        )

    st.divider()

    st.markdown(
        "### What the recommendation engine looks at"
    )

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:
        st.markdown("#### Ingredient intelligence")
        st.markdown(
            """
<div style="
background:rgba(255,255,255,0.55);
padding:20px;
border-radius:18px;
display:flex;
flex-wrap:wrap;
gap:10px;
">
<span style="background:#efe4de;padding:7px 12px;border-radius:999px;color:#332d2b;">Humectants</span>
<span style="background:#efe4de;padding:7px 12px;border-radius:999px;color:#332d2b;">Emollients</span>
<span style="background:#efe4de;padding:7px 12px;border-radius:999px;color:#332d2b;">Antioxidants</span>
<span style="background:#efe4de;padding:7px 12px;border-radius:999px;color:#332d2b;">Skin protection</span>
<span style="background:#efe4de;padding:7px 12px;border-radius:999px;color:#332d2b;">Irritancy</span>
<span style="background:#efe4de;padding:7px 12px;border-radius:999px;color:#332d2b;">Pore-clogging signal</span>
</div>
            """,
            unsafe_allow_html=True
        )

    with feature_col2:
        st.markdown("#### Personalization signals")
        st.markdown(
            """
<div style="
background:rgba(255,255,255,0.55);
padding:20px;
border-radius:18px;
display:flex;
flex-wrap:wrap;
gap:10px;
">
<span style="background:#efe4de;padding:7px 12px;border-radius:999px;color:#332d2b;">Skin type</span>
<span style="background:#efe4de;padding:7px 12px;border-radius:999px;color:#332d2b;">Sensitivity</span>
<span style="background:#efe4de;padding:7px 12px;border-radius:999px;color:#332d2b;">Acne tendency</span>
<span style="background:#efe4de;padding:7px 12px;border-radius:999px;color:#332d2b;">Hydration</span>
<span style="background:#efe4de;padding:7px 12px;border-radius:999px;color:#332d2b;">Barrier support</span>
<span style="background:#efe4de;padding:7px 12px;border-radius:999px;color:#332d2b;">Antioxidant support</span>
</div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown(
        "### Ingredient profile across the product database"
    )

    chart_data = pd.DataFrame(
        {
            "Ingredient group": [
                "Hydration",
                "Emollient",
                "Antioxidant",
                "Skin protection"
            ],
            "Average ingredient count": [
                profiles[
                    "humectant_count"
                ].mean(),
                profiles[
                    "emollient_count"
                ].mean(),
                profiles[
                    "antioxidant_count"
                ].mean(),
                profiles[
                    "skin_protecting_count"
                ].mean()
            ]
        }
    )

    chart = (
        alt.Chart(chart_data)
        .mark_bar(
            color="#a7776a",
            cornerRadiusTopLeft=6,
            cornerRadiusTopRight=6
        )
        .encode(
            x=alt.X(
                "Ingredient group:N",
                title=None,
                sort=[
                    "Hydration",
                    "Emollient",
                    "Antioxidant",
                    "Skin protection"
                ]
            ),
            y=alt.Y(
                "Average ingredient count:Q",
                title="Average ingredient count",
                scale=alt.Scale(
                    zero=True
                )
            ),
            tooltip=[
                "Ingredient group:N",
                alt.Tooltip(
                    "Average ingredient count:Q",
                    format=".2f"
                )
            ]
        )
        .properties(
            height=360
        )
    )

    st.altair_chart(
        chart,
        use_container_width=True
    )

    st.caption(
        "Average number of selected functional ingredient groups "
        "identified across the SkinSmart product dataset."
    )

    st.divider()

    st.markdown(
        "### Recommendation pipeline"
    )

    pipeline1, pipeline2, pipeline3, pipeline4 = st.columns(4)

    with pipeline1:

        st.markdown("### 01")
        st.markdown(
            "**User profile**"
        )
        st.caption(
            "Skin type, sensitivity, acne tendency, goals and hard preferences"
        )

    with pipeline2:

        st.markdown("### 02")
        st.markdown(
            "**Explainable engine · 70%**"
        )
        st.caption(
            "Normalized skincare features create transparent profile-fit scores"
        )

    with pipeline3:

        st.markdown("### 03")
        st.markdown(
            "**ML similarity · 30%**"
        )
        st.caption(
            "StandardScaler + Nearest Neighbors measure similarity in feature space"
        )

    with pipeline4:

        st.markdown("### 04")
        st.markdown(
            "**Hybrid ranking**"
        )
        st.caption(
            "Both signals are combined into the final personalized ranking"
        )

    st.divider()

    with st.expander(
        "Technical score diagnostics",
        expanded=False
    ):
        st.caption(
            "Internal dataset diagnostics used during feature engineering. "
            "These values are not consumer-facing match percentages."
        )

        score1, score2, score3 = st.columns(3)

        with score1:
            st.metric(
                "Average dry-skin score",
                f"{profiles['dry_skin_score'].mean():.1f}"
            )

        with score2:
            st.metric(
                "Average sensitive-skin score",
                f"{profiles['sensitive_skin_score'].mean():.1f}"
            )

        with score3:
            st.metric(
                "Average acne/oily score",
                f"{profiles['acne_oily_score'].mean():.1f}"
            )

        st.caption(
            "A negative internal score does not mean a product is harmful. "
            "It indicates that, under the engineered scoring rules, some "
            "characteristics reduce suitability for that skin profile."
        )

    st.divider()

    st.markdown(
        "### How personalization changes the results"
    )

    st.write(
        "To demonstrate that SkinSmart AI does not return the same "
        "products for every user, two different skin profiles are "
        "compared below."
    )

    dry_sensitive_profile = {
        "skin_type": "dry",
        "sensitive": True,
        "acne_prone": False,
        "fragrance_free": True,
        "avoid_drying_alcohol": True,
        "max_comedogenicity": 2,
        "concerns": [
            "hydration",
            "barrier_support"
        ]
    }

    oily_acne_profile = {
        "skin_type": "oily",
        "sensitive": False,
        "acne_prone": True,
        "fragrance_free": False,
        "avoid_drying_alcohol": True,
        "max_comedogenicity": 1,
        "concerns": [
            "antioxidant_support"
        ]
    }

    dry_results = get_app_recommendations(
        profiles,
        dry_sensitive_profile,
        top_n=3
    )

    oily_results = get_app_recommendations(
        profiles,
        oily_acne_profile,
        top_n=3
    )

    compare1, compare2 = st.columns(2)

    with compare1:

        st.markdown(
            "#### 💧 Dry + sensitive profile"
        )

        st.caption(
            "Hydration • Barrier support • Fragrance-free"
        )

        for rank, (_, row) in enumerate(
            dry_results.iterrows(),
            start=1
        ):

            with st.container(
                border=True
            ):

                st.markdown(
                    f"**#{rank} · "
                    f"{row['product_name']}**"
                )

                st.caption(
                    f"{row['brand']} · "
                    f"Score "
                    f"{row['personalized_score']:.2f}"
                )

    with compare2:

        st.markdown(
            "#### ✨ Oily + acne-prone profile"
        )

        st.caption(
            "Acne-conscious • Antioxidant support"
        )

        for rank, (_, row) in enumerate(
            oily_results.iterrows(),
            start=1
        ):

            with st.container(
                border=True
            ):

                st.markdown(
                    f"**#{rank} · "
                    f"{row['product_name']}**"
                )

                st.caption(
                    f"{row['brand']} · "
                    f"Score "
                    f"{row['personalized_score']:.2f}"
                )

    st.caption(
        "Different user profiles change both the product scores "
        "and the final ranking."
    )

    st.divider()

    st.markdown(
        "### Evaluation & validation"
    )

    st.write(
        "The deployed hybrid recommender was tested across several "
        "contrasting user profiles to check whether rankings genuinely "
        "change and whether the hard preference filters behave as intended."
    )

    evaluation_profiles = {
        "Dry + Sensitive": {
            "skin_type": "dry",
            "sensitive": True,
            "acne_prone": False,
            "fragrance_free": True,
            "avoid_drying_alcohol": True,
            "max_comedogenicity": 2,
            "concerns": ["hydration", "barrier_support"]
        },
        "Dry + Acne-prone": {
            "skin_type": "dry",
            "sensitive": False,
            "acne_prone": True,
            "fragrance_free": False,
            "avoid_drying_alcohol": True,
            "max_comedogenicity": 1,
            "concerns": ["hydration"]
        },
        "Oily + Acne-prone": {
            "skin_type": "oily",
            "sensitive": False,
            "acne_prone": True,
            "fragrance_free": False,
            "avoid_drying_alcohol": True,
            "max_comedogenicity": 1,
            "concerns": ["antioxidant_support"]
        },
        "Oily + Sensitive": {
            "skin_type": "oily",
            "sensitive": True,
            "acne_prone": False,
            "fragrance_free": True,
            "avoid_drying_alcohol": True,
            "max_comedogenicity": 2,
            "concerns": ["barrier_support"]
        },
        "Normal + Hydration": {
            "skin_type": "normal",
            "sensitive": False,
            "acne_prone": False,
            "fragrance_free": False,
            "avoid_drying_alcohol": False,
            "max_comedogenicity": None,
            "concerns": ["hydration"]
        },
        "Normal + Antioxidants": {
            "skin_type": "normal",
            "sensitive": False,
            "acne_prone": False,
            "fragrance_free": False,
            "avoid_drying_alcohol": False,
            "max_comedogenicity": None,
            "concerns": ["antioxidant_support"]
        }
    }

    evaluation_results = {}

    for profile_name, profile in evaluation_profiles.items():
        result = recommend_products_hybrid(
            profiles,
            profile,
            top_n=5
        )

        evaluation_results[profile_name] = list(
            result["product_name"]
        )

    profile_names = list(evaluation_results.keys())
    overlap_values = []

    for i in range(len(profile_names)):
        for j in range(i + 1, len(profile_names)):
            products_a = set(
                evaluation_results[profile_names[i]]
            )
            products_b = set(
                evaluation_results[profile_names[j]]
            )

            shared = len(
                products_a.intersection(products_b)
            )

            overlap_values.append(
                (shared / 5) * 100
            )

    median_overlap = (
        pd.Series(overlap_values).median()
        if overlap_values
        else 0
    )

    eval1, eval2, eval3 = st.columns(3)

    with eval1:
        st.metric(
            "Profiles evaluated",
            len(evaluation_profiles)
        )

    with eval2:
        st.metric(
            "Median Top-5 overlap",
            f"{median_overlap:.0f}%"
        )

    filter_test_profile = {
        "skin_type": "dry",
        "sensitive": True,
        "acne_prone": False,
        "fragrance_free": True,
        "avoid_drying_alcohol": True,
        "max_comedogenicity": 1,
        "concerns": ["hydration"]
    }

    filter_results = recommend_products_hybrid(
        profiles,
        filter_test_profile,
        top_n=20
    )

    fragrance_violations = int(
        filter_results["contains_fragrance"].sum()
    )

    alcohol_violations = int(
        filter_results["contains_drying_alcohol"].sum()
    )

    comedogenicity_violations = int(
        (
            filter_results["max_comedogenicity_filled"]
            > 1
        ).sum()
    )

    total_filter_violations = (
        fragrance_violations
        + alcohol_violations
        + comedogenicity_violations
    )

    with eval3:
        st.metric(
            "Hard-filter violations",
            total_filter_violations
        )

    st.caption(
        "A 20% median Top-5 overlap means the tested profiles typically "
        "received substantially different recommendation sets."
    )

    st.markdown(
        "#### Preference-filter validation"
    )

    if total_filter_violations == 0:
        st.success(
            "Validation passed: the restrictive test returned no fragranced "
            "products, no products with drying alcohol, and no products above "
            "the selected pore-clogging threshold."
        )
    else:
        st.warning(
            "One or more filter violations were detected. "
            "Review the filtering logic before deployment."
        )

        st.markdown(
        "#### RAG assistant evaluation"
    )

    st.write(
        "The Ask SkinSmart assistant was evaluated both quantitatively "
        "and with controlled user-profile grounding tests."
    )

    rag1, rag2, rag3, rag4 = st.columns(4)

    with rag1:
        st.metric(
            "Faithfulness",
            "0.85"
        )

    with rag2:
        st.metric(
            "Answer correctness",
            "0.76"
        )

    with rag3:
        st.metric(
            "Context precision",
            "1.00"
        )

    with rag4:
        st.metric(
            "Context recall",
            "0.86"
        )

    st.caption(
        "RAG evaluation measured whether generated answers remain grounded "
        "in retrieved SkinSmart knowledge and whether the relevant context "
        "is successfully retrieved."
    )

    st.markdown(
        "##### User-profile grounding tests"
    )

    grounding_tests = pd.DataFrame(
        {
            "Test": [
                "Dry skin + hydration",
                "Fragrance-free not selected",
                "Sensitive skin = False",
                "Sensitive skin = True"
            ],
            "Expected behavior": [
                "Use dry-skin and hydration information",
                "Do not claim fragrance-free was required",
                "Do not describe the user as sensitive",
                "Use sensitivity in the explanation"
            ],
            "Result": [
                "Pass",
                "Pass",
                "Pass",
                "Pass"
            ]
        }
    )

    st.dataframe(
        grounding_tests,
        hide_index=True,
        use_container_width=True
    )

    st.success(
        "All 4 controlled profile-grounding tests passed."
    )

    st.caption(
        "The tests check that the assistant distinguishes between user "
        "preferences and product characteristics instead of inventing "
        "profile attributes."
    )

    st.divider()

    st.markdown(
        "#### Outlier diagnostic"
    )

    pdrn_rows = profiles[
        profiles["product_name"] == "Pdrn Booster"
    ]

    if not pdrn_rows.empty:

        pdrn = pdrn_rows.iloc[0]

        diagnostic_columns = [
            "dry_skin_score",
            "acne_oily_score",
            "sensitive_skin_score",
            "humectant_count",
            "skin_protecting_count",
            "antioxidant_count"
        ]

        high_percentile_count = 0

        for column in diagnostic_columns:
            percentile = (
                profiles[column]
                .rank(pct=True)
                .loc[pdrn.name]
                * 100
            )

            if percentile >= 99:
                high_percentile_count += 1

        st.info(
            f"Pdrn Booster ranks at or above the 99th percentile in "
            f"{high_percentile_count} of {len(diagnostic_columns)} major "
            "recommendation signals. This explains why it frequently appears "
            "near the top across different profiles without manually boosting "
            "its score."
        )

    st.divider()

    st.markdown(
    "### Current limitations"
)

    st.info(
        "SkinSmart AI combines engineered skincare features, machine-learning "
        "similarity, and a retrieval-augmented AI assistant. The system is not "
        "trained on clinical outcomes, purchases, or real user feedback. "
        "The RAG assistant can also occasionally miss relevant context during "
        "retrieval, so generated explanations should be treated as informative "
        "rather than clinical guidance. Recommendations are personalized "
        "ingredient-data guidance, not medical advice."
    )

    st.markdown(
        "### Future development"
    )

    future1, future2, future3, future4 = st.columns(4)

    future1.markdown(
        "#### 👤 User feedback"
    )
    future1.write(
        "Learn from real user ratings and product experiences."
    )

    future2.markdown(
        "#### 📊 Larger dataset"
    )
    future2.write(
        "Expand the product and ingredient database across more brands."
    )

    future3.markdown(
        "#### 💬 Better retrieval"
    )
    future3.write(
        "Improve retrieval coverage and expand the assistant's skincare knowledge base."
    )

    future4.markdown(
        "#### 🧪 Expert validation"
    )
    future4.write(
        "Compare recommendation logic with dermatologist or skincare-expert guidance."
    )

# ------------------------------------------------------------
# Ask SkinSmart - RAG Chatbot
# ------------------------------------------------------------

if page == "Ask SkinSmart":

    

    st.markdown("## 💬 Ask SkinSmart")

    st.markdown(
    """
    <p style="color:#6f6661; font-size:1rem;">
        Ask questions about SkinSmart recommendations, ingredients,
        product characteristics, and how the system works.
    </p>
    """,
    unsafe_allow_html=True,
    )

    try:
        rag_engine = load_rag_engine()
    except ValueError as exc:
        if "GROQ_API_KEY" not in str(exc):
            raise
        st.info("AI chat is temporarily unavailable in this demo. You can still explore the recommendations and model insights.")
        st.stop()

    # Create chat history once
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    if st.button("Clear chat"):
        st.session_state.rag_messages = []
        st.rerun()

    # Show previous messages
    for message in st.session_state.rag_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_question = st.chat_input(
        "Ask SkinSmart a question..."
    )

    if user_question:

        st.session_state.rag_messages.append(
            {
                 "role": "user",
                "content": user_question,
            }
        )

        with st.chat_message("user"):
             st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("SkinSmart is thinking..."):

                profile_context = st.session_state.get("user_profile", {})
                recommendations_context = st.session_state.get("recommendations")

            if recommendations_context is not None:
                recommendations_text = recommendations_context.to_string(index=False)
            else:
                recommendations_text = "No recommendations have been generated yet."

            enhanced_question = f"""
You are SkinSmart AI, an explainable skincare recommendation assistant.

CURRENT USER PROFILE:
{profile_context}

CURRENT RECOMMENDATIONS:
{recommendations_text}

USER QUESTION:
{user_question}

IMPORTANT RULES:

1. Treat the CURRENT USER PROFILE as the only source of truth about the user.
2. Do not infer that the user is sensitive, acne-prone, fragrance-sensitive,
   or has any other trait unless it is explicitly true in the user profile.
3. Do not confuse product characteristics with user characteristics.
4. Explain recommendations using only available information.
5. If a preference is False, say that it was not selected as a requirement.
6. If information is unavailable, say so rather than guessing.
7. Do not provide medical diagnosis or treatment advice.
8. Keep explanations clear, personalized, and concise.
9. Never expose internal field names, raw dictionary values, or implementation details such as
   "fragrance_free: False" or "sensitive: False". Translate them into natural language.

10. When answering about a user preference, say things like
    "You did not select fragrance-free as a requirement"
    instead of showing the underlying variable or boolean value.
"""

            rag_response = rag_engine.query(enhanced_question)
            answer = str(rag_response)

        st.markdown(answer)

        st.session_state.rag_messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
         )

st.markdown("---")



# --------------------------------------------------
# Disclaimer
# --------------------------------------------------

st.markdown(
    """
<div class="disclaimer">
<strong>Portfolio demonstration:</strong> This version uses 80 fictional skincare product profiles with synthetic features and illustrative scores. Results are not recommendations of real products.
<p>The datasets credited below were used during the original project's development, not as product records in this public demonstration.</p>
SkinSmart AI recommendations are based on ingredient-data patterns
and user-selected preferences. They are intended for informational
purposes and are not a medical diagnosis or treatment recommendation.
<p>
Data source:
<a href="https://huggingface.co/datasets/Ichlibitiche/incidb-skincare-free-sample">
INCIDB Skincare & Cosmetics free sample
</a> (ODbL 1.0).
Data adapted for the SkinSmart AI project.
</p>
<p>
Recommendation data:
<a href="https://huggingface.co/datasets/thebeautyapi/beautyproducts">
The Beauty API
</a>
(<a href="https://creativecommons.org/licenses/by-nc/4.0/">
CC BY-NC 4.0
</a>).
Data processed and adapted for SkinSmart AI.
</p>
</div>
    """,
    unsafe_allow_html=True
)