from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pandas as pd


# --------------------------------------------------
# Version 2 configuration
# --------------------------------------------------

V2_WEIGHTS = {
    "dry_skin": 1.0,
    "oily_skin": 1.0,
    "sensitive_skin": 1.0,
    "acne_prone": 0.5,
    "hydration": 1.0,
    "barrier_support": 1.0,
    "antioxidant_support": 1.0
}


# --------------------------------------------------
# Hybrid recommendation configuration
# --------------------------------------------------

HYBRID_V2_WEIGHT = 0.70
HYBRID_ML_WEIGHT = 0.30

ML_FEATURES = [
    "dry_skin_score",
    "sensitive_skin_score",
    "acne_oily_score",
    "humectant_count",
    "emollient_count",
    "antioxidant_count",
    "skin_protecting_count",
    "max_irritancy_filled",
    "max_comedogenicity_filled"
]


def normalize_series(series):
    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return series * 0

    return (series - minimum) / (maximum - minimum)


def validate_user_profile(user_profile):
    valid_skin_types = {"normal", "dry", "oily"}

    if user_profile.get("skin_type") not in valid_skin_types:
        raise ValueError(
            "skin_type must be one of: normal, dry, oily"
        )

    for key in [
        "sensitive",
        "acne_prone",
        "fragrance_free",
        "avoid_drying_alcohol"
    ]:
        if not isinstance(user_profile.get(key), bool):
            raise ValueError(
                f"{key} must be True or False"
            )

    max_comedogenicity = user_profile.get(
        "max_comedogenicity"
    )

    if max_comedogenicity is not None:
        if max_comedogenicity not in [0, 1, 2, 3, 4, 5]:
            raise ValueError(
                "max_comedogenicity must be None or 0–5"
            )


# --------------------------------------------------
# Original Version 1 recommender
# --------------------------------------------------

def recommend_products(profiles, user_profile, top_n=10):
    results = profiles.copy()

    results["personalized_score"] = 0.0

    if user_profile["skin_type"] == "dry":
        results["personalized_score"] += results["dry_skin_score"]

    elif user_profile["skin_type"] == "oily":
        results["personalized_score"] += results["acne_oily_score"]

    if user_profile["sensitive"]:
        results["personalized_score"] += results["sensitive_skin_score"]

    if user_profile["acne_prone"]:
        results["personalized_score"] += (
            0.5 * results["acne_oily_score"]
        )

    for concern in user_profile.get("concerns", []):

        if concern == "hydration":
            results["personalized_score"] += (
                2 * results["humectant_count"]
            )

        elif concern == "barrier_support":
            results["personalized_score"] += (
                results["emollient_count"]
                + 2 * results["skin_protecting_count"]
            )

        elif concern == "antioxidant_support":
            results["personalized_score"] += (
                2 * results["antioxidant_count"]
            )

    if user_profile["fragrance_free"]:
        results = results[
            results["contains_fragrance"] == False
        ]

    if user_profile["avoid_drying_alcohol"]:
        results = results[
            results["contains_drying_alcohol"] == False
        ]

    if user_profile["max_comedogenicity"] is not None:
        results = results[
            results["max_comedogenicity_filled"]
            <= user_profile["max_comedogenicity"]
        ]

    return (
        results
        .sort_values("personalized_score", ascending=False)
        .head(top_n)
    )


# --------------------------------------------------
# Version 2 explainable recommender
# --------------------------------------------------

def recommend_products_v2(
    profiles,
    user_profile,
    top_n=10
):
    validate_user_profile(user_profile)

    results = profiles.copy()

    results["norm_dry_skin"] = normalize_series(
        results["dry_skin_score"]
    )

    results["norm_acne_oily"] = normalize_series(
        results["acne_oily_score"]
    )

    results["norm_sensitive"] = normalize_series(
        results["sensitive_skin_score"]
    )

    results["norm_humectants"] = normalize_series(
        results["humectant_count"]
    )

    results["norm_emollients"] = normalize_series(
        results["emollient_count"]
    )

    results["norm_skin_protecting"] = normalize_series(
        results["skin_protecting_count"]
    )

    results["norm_antioxidants"] = normalize_series(
        results["antioxidant_count"]
    )

    results["skin_type_contribution"] = 0.0
    results["sensitivity_contribution"] = 0.0
    results["acne_contribution"] = 0.0
    results["hydration_contribution"] = 0.0
    results["barrier_contribution"] = 0.0
    results["antioxidant_contribution"] = 0.0

    if user_profile["skin_type"] == "dry":
        results["skin_type_contribution"] = (
            V2_WEIGHTS["dry_skin"]
            * results["norm_dry_skin"]
        )

    elif user_profile["skin_type"] == "oily":
        results["skin_type_contribution"] = (
            V2_WEIGHTS["oily_skin"]
            * results["norm_acne_oily"]
        )

    if user_profile["sensitive"]:
        results["sensitivity_contribution"] = (
            V2_WEIGHTS["sensitive_skin"]
            * results["norm_sensitive"]
        )

    # Avoid double-counting acne/oily when oily skin is already selected.
    if (
        user_profile["acne_prone"]
        and user_profile["skin_type"] != "oily"
    ):
        results["acne_contribution"] = (
            V2_WEIGHTS["acne_prone"]
            * results["norm_acne_oily"]
        )

    concerns = user_profile.get(
        "concerns",
        []
    )

    if "hydration" in concerns:
        results["hydration_contribution"] = (
            V2_WEIGHTS["hydration"]
            * results["norm_humectants"]
        )

    if "barrier_support" in concerns:
        barrier_score = (
            results["norm_emollients"]
            + results["norm_skin_protecting"]
        ) / 2

        results["barrier_contribution"] = (
            V2_WEIGHTS["barrier_support"]
            * barrier_score
        )

    if "antioxidant_support" in concerns:
        results["antioxidant_contribution"] = (
            V2_WEIGHTS["antioxidant_support"]
            * results["norm_antioxidants"]
        )

    results["personalized_score_v2"] = (
        results["skin_type_contribution"]
        + results["sensitivity_contribution"]
        + results["acne_contribution"]
        + results["hydration_contribution"]
        + results["barrier_contribution"]
        + results["antioxidant_contribution"]
    )

    if user_profile["fragrance_free"]:
        results = results[
            results["contains_fragrance"] == False
        ]

    if user_profile["avoid_drying_alcohol"]:
        results = results[
            results["contains_drying_alcohol"] == False
        ]

    if user_profile["max_comedogenicity"] is not None:
        results = results[
            results["max_comedogenicity_filled"]
            <= user_profile["max_comedogenicity"]
        ]

    return (
        results
        .sort_values(
            "personalized_score_v2",
            ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )


# --------------------------------------------------
# Machine-learning layer
# --------------------------------------------------

def create_user_vector(user_profile, profiles):
    """
    Build a target feature vector from the selected skin profile.
    Dataset averages are used as the neutral baseline, while relevant
    user preferences move selected features toward high/low quantiles.
    """
    validate_user_profile(user_profile)

    target = {
        feature: profiles[feature].mean()
        for feature in ML_FEATURES
    }

    if user_profile["skin_type"] == "dry":
        target["dry_skin_score"] = (
            profiles["dry_skin_score"].quantile(0.90)
        )

    elif user_profile["skin_type"] == "oily":
        target["acne_oily_score"] = (
            profiles["acne_oily_score"].quantile(0.90)
        )

    if user_profile["sensitive"]:
        target["sensitive_skin_score"] = (
            profiles["sensitive_skin_score"].quantile(0.90)
        )
        target["max_irritancy_filled"] = (
            profiles["max_irritancy_filled"].quantile(0.10)
        )

    if user_profile["acne_prone"]:
        target["acne_oily_score"] = (
            profiles["acne_oily_score"].quantile(0.90)
        )
        target["max_comedogenicity_filled"] = (
            profiles["max_comedogenicity_filled"].quantile(0.10)
        )

    concerns = user_profile.get("concerns", [])

    if "hydration" in concerns:
        target["humectant_count"] = (
            profiles["humectant_count"].quantile(0.90)
        )

    if "barrier_support" in concerns:
        target["emollient_count"] = (
            profiles["emollient_count"].quantile(0.90)
        )
        target["skin_protecting_count"] = (
            profiles["skin_protecting_count"].quantile(0.90)
        )

    if "antioxidant_support" in concerns:
        target["antioxidant_count"] = (
            profiles["antioxidant_count"].quantile(0.90)
        )

    return pd.DataFrame(
        [[target[feature] for feature in ML_FEATURES]],
        columns=ML_FEATURES
    )


def calculate_ml_similarity(profiles, user_profile):
    """
    Fit a StandardScaler + NearestNeighbors feature-space model and
    calculate a similarity score for every product.
    """
    X_products = profiles[ML_FEATURES].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_products)

    model = NearestNeighbors(
        n_neighbors=len(profiles),
        metric="euclidean"
    )
    model.fit(X_scaled)

    user_vector = create_user_vector(
        user_profile,
        profiles
    )
    user_scaled = scaler.transform(user_vector)

    distances, indices = model.kneighbors(
        user_scaled,
        n_neighbors=len(profiles)
    )

    ml_scores = pd.DataFrame({
        "product_index": indices[0],
        "ml_distance": distances[0]
    })

    ml_scores["ml_similarity"] = (
        1 / (1 + ml_scores["ml_distance"])
    )

    ml_scores["product_id"] = (
        profiles.iloc[
            ml_scores["product_index"]
        ]["product_id"].values
    )

    return ml_scores[
        [
            "product_id",
            "ml_distance",
            "ml_similarity"
        ]
    ]


def recommend_products_ml(
    profiles,
    user_profile,
    top_n=5
):
    """
    Pure ML/content-similarity recommendations for diagnostics
    and model comparison.
    """
    validate_user_profile(user_profile)

    ml_scores = calculate_ml_similarity(
        profiles,
        user_profile
    )

    results = profiles.merge(
        ml_scores,
        on="product_id",
        how="left"
    )

    # Apply the same hard preferences used by V2.
    if user_profile["fragrance_free"]:
        results = results[
            results["contains_fragrance"] == False
        ]

    if user_profile["avoid_drying_alcohol"]:
        results = results[
            results["contains_drying_alcohol"] == False
        ]

    if user_profile["max_comedogenicity"] is not None:
        results = results[
            results["max_comedogenicity_filled"]
            <= user_profile["max_comedogenicity"]
        ]

    return (
        results
        .sort_values("ml_similarity", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# --------------------------------------------------
# Hybrid recommender: 70% explainable V2 + 30% ML
# --------------------------------------------------

def recommend_products_hybrid(
    profiles,
    user_profile,
    top_n=5,
    v2_weight=HYBRID_V2_WEIGHT,
    ml_weight=HYBRID_ML_WEIGHT
):
    validate_user_profile(user_profile)

    if not 0 <= v2_weight <= 1:
        raise ValueError("v2_weight must be between 0 and 1")

    if not 0 <= ml_weight <= 1:
        raise ValueError("ml_weight must be between 0 and 1")

    if abs((v2_weight + ml_weight) - 1.0) > 1e-9:
        raise ValueError(
            "v2_weight and ml_weight must add up to 1"
        )

    # V2 applies the hard user filters before the hybrid merge.
    v2_results = recommend_products_v2(
        profiles,
        user_profile,
        top_n=len(profiles)
    ).copy()

    if v2_results.empty:
        return v2_results

    ml_scores = calculate_ml_similarity(
        profiles,
        user_profile
    )

    hybrid = v2_results.merge(
        ml_scores,
        on="product_id",
        how="left"
    )

    min_v2 = hybrid["personalized_score_v2"].min()
    max_v2 = hybrid["personalized_score_v2"].max()

    if max_v2 == min_v2:
        hybrid["v2_normalized"] = 0.0
    else:
        hybrid["v2_normalized"] = (
            hybrid["personalized_score_v2"] - min_v2
        ) / (
            max_v2 - min_v2
        )

    min_ml = hybrid["ml_similarity"].min()
    max_ml = hybrid["ml_similarity"].max()

    if max_ml == min_ml:
        hybrid["ml_normalized"] = 0.0
    else:
        hybrid["ml_normalized"] = (
            hybrid["ml_similarity"] - min_ml
        ) / (
            max_ml - min_ml
        )

    hybrid["v2_weighted_contribution"] = (
        v2_weight * hybrid["v2_normalized"]
    )

    hybrid["ml_weighted_contribution"] = (
        ml_weight * hybrid["ml_normalized"]
    )

    hybrid["hybrid_score"] = (
        hybrid["v2_weighted_contribution"]
        + hybrid["ml_weighted_contribution"]
    )

    return (
        hybrid
        .sort_values(
            "hybrid_score",
            ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )


# --------------------------------------------------
# Recommendation explanations
# --------------------------------------------------

def explain_recommendation(row, user_profile):
    reasons = []

    if user_profile["skin_type"] == "dry":
        if row["humectant_count"] >= 5:
            reasons.append("strong hydration support")

        if row["emollient_count"] > 0:
            reasons.append("helps support softness")

    if user_profile["sensitive"]:
        if row["max_irritancy_filled"] <= 1:
            reasons.append("low irritancy profile")

        if not row["contains_fragrance"]:
            reasons.append("fragrance-free")

        if not row["contains_drying_alcohol"]:
            reasons.append("no drying alcohol")

    if user_profile["acne_prone"]:
        if row["max_comedogenicity_filled"] <= 1:
            reasons.append("low comedogenicity")

    for concern in user_profile.get("concerns", []):

        if (
            concern == "hydration"
            and row["humectant_count"] >= 5
        ):
            reasons.append("matches your hydration goal")

        elif concern == "barrier_support":
            if (
                row["emollient_count"] > 0
                or row["skin_protecting_count"] > 0
            ):
                reasons.append("supports your skin barrier")

        elif concern == "antioxidant_support":
            if row["antioxidant_count"] > 0:
                reasons.append("antioxidant support")

    return "; ".join(dict.fromkeys(reasons))


# --------------------------------------------------
# App-facing recommendation function
# --------------------------------------------------

def get_app_recommendations(
    profiles,
    user_profile,
    top_n=5
):
    """
    Main app interface.

    The visible ranking is produced by the hybrid recommender:
    70% explainable V2 score + 30% ML nearest-neighbor similarity.
    """
    results = recommend_products_hybrid(
        profiles,
        user_profile,
        top_n=top_n
    ).copy()

    if results.empty:
        return results

    results["why_recommended"] = results.apply(
        lambda row: explain_recommendation(
            row,
            user_profile
        ),
        axis=1
    )

    # Keep the existing app column name for backward compatibility.
    results["personalized_score"] = (
        results["hybrid_score"]
    )

    return results[
        [
            "product_id",
            "brand",
            "product_name",

            "personalized_score",
            "hybrid_score",
            "personalized_score_v2",
            "ml_similarity",
            "v2_normalized",
            "ml_normalized",
            "v2_weighted_contribution",
            "ml_weighted_contribution",

            "skin_type_contribution",
            "sensitivity_contribution",
            "acne_contribution",
            "hydration_contribution",
            "barrier_contribution",
            "antioxidant_contribution",

            "humectant_count",
            "skin_protecting_count",
            "emollient_count",

            "why_recommended"
        ]
    ].reset_index(drop=True)