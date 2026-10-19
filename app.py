import json
import pickle
import warnings
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# Paths are resolved relative to this file so the app works no matter which
# folder Streamlit is launched from (important on Streamlit Community Cloud).
BASE_DIR = Path(__file__).resolve().parent

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Soil Liquefaction Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colors shared by the CSS and the charts
RISK_COLOR = "#B4532A"   # pushes toward liquefaction
SAFE_COLOR = "#2F6F62"   # pushes toward stability
MUTED = "#8A8079"

# Display metadata for each model input (keys must match feature_columns.json)
FEATURES = {
    "Converted_SPT_N1_60": dict(
        label="SPT N1(60)", unit="blows", default=15.0, min=0.0, max=100.0, step=1.0, fmt="%.1f",
        help="Standard Penetration Test blow count corrected for overburden pressure and hammer energy.",
    ),
    "Fines_Content": dict(
        label="Fines content", unit="%", default=10.0, min=0.0, max=100.0, step=0.5, fmt="%.1f",
        help="Percentage of soil passing the No. 200 sieve (0.075 mm).",
    ),
    "D50": dict(
        label="Median grain size D50", unit="mm", default=0.30, min=0.0, max=10.0, step=0.01, fmt="%.3f",
        help="Grain diameter at which 50% of the soil by weight is finer.",
    ),
    "Effective_Stress": dict(
        label="Vertical effective stress σ'v", unit="kPa", default=50.0, min=0.0, max=1000.0, step=1.0, fmt="%.1f",
        help="Effective vertical overburden stress at the depth of interest.",
    ),
    "Calculated_CSR": dict(
        label="Cyclic stress ratio (CSR)", unit="", default=0.17, min=0.0, max=2.0, step=0.005, fmt="%.3f",
        help="Cyclic Stress Ratio induced by the design earthquake at the depth of interest.",
    ),
}

# --- CUSTOM CSS ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap');

    .block-container {{ padding-top: 2.2rem; max-width: 1200px; }}

    .eyebrow {{
        text-transform: uppercase; letter-spacing: .14em; font-size: .78rem;
        font-weight: 600; color: {RISK_COLOR}; margin-bottom: .2rem;
    }}
    .app-title {{
        font-family: 'Source Serif 4', Georgia, serif; font-weight: 700;
        font-size: 2.6rem; line-height: 1.1; margin: 0 0 .4rem 0;
    }}
    .app-sub {{ color: #6B625B; font-size: 1.05rem; max-width: 720px; margin-bottom: 1.6rem; }}

    .verdict {{
        border-radius: 14px; padding: 1.4rem 1.6rem; border: 1px solid;
        height: 100%;
    }}
    .verdict.danger {{ background: #FBEDE6; border-color: #EBC3AF; }}
    .verdict.safe   {{ background: #E8F2EE; border-color: #B9D6CB; }}
    .verdict .label {{ font-size: .8rem; text-transform: uppercase; letter-spacing: .12em; font-weight: 600; }}
    .verdict.danger .label, .verdict.danger .headline {{ color: #8C3A17; }}
    .verdict.safe .label,   .verdict.safe .headline   {{ color: #1F5246; }}
    .verdict .headline {{
        font-family: 'Source Serif 4', Georgia, serif; font-size: 2rem; font-weight: 700;
        margin: .25rem 0 .5rem 0; line-height: 1.15;
    }}
    .verdict p {{ color: #4A423C; margin: 0; }}

    .meter {{ margin-top: 1.1rem; }}
    .meter-track {{
        position: relative; height: 10px; border-radius: 6px;
        background: linear-gradient(90deg, {SAFE_COLOR} 0%, #D9CFA0 50%, {RISK_COLOR} 100%);
    }}
    .meter-mark {{
        position: absolute; top: -5px; width: 4px; height: 20px; border-radius: 2px;
        background: #2B2622; transform: translateX(-2px);
    }}
    .meter-mid {{ position: absolute; top: -3px; left: 50%; width: 1px; height: 16px; background: rgba(0,0,0,.35); }}
    .meter-legend {{ display: flex; justify-content: space-between; font-size: .78rem; color: #6B625B; margin-top: .35rem; }}

    .footer {{ text-align: center; color: {MUTED}; font-size: .85rem; margin-top: 2.5rem; }}
    [data-testid="stSidebar"] h2 {{ font-family: 'Source Serif 4', Georgia, serif; }}
</style>
""", unsafe_allow_html=True)


# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    required = ["best_model.pkl", "scaler.pkl", "feature_columns.json", "best_threshold.json"]
    missing = [f for f in required if not (BASE_DIR / f).exists()]
    if missing:
        return None, f"Missing files: {', '.join(missing)}. Make sure they are uploaded next to app.py."

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(BASE_DIR / "best_model.pkl", "rb") as f:
                model = pickle.load(f)
            with open(BASE_DIR / "scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
        with open(BASE_DIR / "feature_columns.json") as f:
            columns = json.load(f)["feature_names"]
        with open(BASE_DIR / "best_threshold.json") as f:
            threshold = json.load(f)["best_threshold"]
    except Exception as e:
        return None, f"Error loading files: {e}"

    # Approximate range of the training data, recovered from the model's support vectors
    sv = scaler.inverse_transform(model.support_vectors_)
    typical = {c: (float(sv[:, i].min()), float(sv[:, i].max())) for i, c in enumerate(columns)}

    return dict(model=model, scaler=scaler, columns=columns, threshold=threshold, typical=typical), "ok"


artifacts, status = load_artifacts()
if artifacts is None:
    st.error(status)
    st.stop()

model = artifacts["model"]
scaler = artifacts["scaler"]
feature_cols = artifacts["columns"]
optimal_threshold = artifacts["threshold"]
typical_range = artifacts["typical"]


def decision_scores(raw: pd.DataFrame) -> np.ndarray:
    """Raw SVC decision scores for inputs given in original (unscaled) units."""
    scaled = scaler.transform(raw[feature_cols])
    if hasattr(model, "predict_proba"):
        return model.predict_proba(scaled)[:, 1]
    return model.decision_function(scaled)


def risk_score(decision):
    """Map the decision score to 0-100 so that 50 sits exactly on the tuned threshold."""
    return 100 / (1 + np.exp(-(np.asarray(decision) - optimal_threshold)))


@st.cache_data(show_spinner=False)
def explain(values: tuple):
    """SHAP values (in decision-score units) for one input, relative to the average training soil."""
    def f(x_scaled):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(x_scaled)[:, 1]
        return model.decision_function(x_scaled)

    x_scaled = scaler.transform(pd.DataFrame([values], columns=feature_cols))
    background = np.zeros((1, len(feature_cols)))  # zeros in scaled space = training-set mean
    explainer = shap.KernelExplainer(f, background)
    sv = explainer.shap_values(x_scaled, nsamples=2 ** len(feature_cols) + 16, silent=True)
    sv = np.asarray(sv[1] if isinstance(sv, list) else sv).reshape(-1)
    base = float(np.asarray(explainer.expected_value).reshape(-1)[-1])
    return sv, base


# --- SIDEBAR INPUTS ---
def reset_inputs():
    for key, meta in FEATURES.items():
        st.session_state[key] = meta["default"]


with st.sidebar:
    st.header("Soil parameters")
    st.caption("Results update as you change a value.")

    user_input = {}
    for key, meta in FEATURES.items():
        st.session_state.setdefault(key, meta["default"])
        label = f"{meta['label']} ({meta['unit']})" if meta["unit"] else meta["label"]
        lo, hi = typical_range[key]
        user_input[key] = st.number_input(
            label,
            min_value=meta["min"], max_value=meta["max"],
            step=meta["step"], format=meta["fmt"], key=key,
            help=f"{meta['help']}  \nTraining data range ≈ {lo:.3g} – {hi:.3g}",
        )

    st.button("Reset to defaults", on_click=reset_inputs, width="stretch")

# --- HEADER ---
st.markdown("<div class='eyebrow'>Geotechnical earthquake engineering · ML</div>", unsafe_allow_html=True)
st.markdown("<div class='app-title'>Soil Liquefaction Predictor</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='app-sub'>Estimate whether a saturated soil layer is likely to liquefy under a given earthquake, "
    "and see which soil properties drive the model's decision.</div>",
    unsafe_allow_html=True,
)

# --- PREDICTION ---
input_df = pd.DataFrame([user_input])[feature_cols]
try:
    decision = float(decision_scores(input_df)[0])
except Exception as e:
    st.error(f"An error occurred during prediction: {e}")
    st.stop()

liquefies = decision >= optimal_threshold
risk = float(risk_score(decision))

out_of_range = [
    FEATURES[c]["label"] for c in feature_cols
    if not (typical_range[c][0] <= user_input[c] <= typical_range[c][1])
]

col_verdict, col_stats = st.columns([1.5, 1], gap="large")

with col_verdict:
    css_class, label, headline, text = (
        ("danger", "Prediction", "Liquefaction is likely",
         "These soil and loading conditions resemble cases that liquefied in the training data. "
         "A detailed site investigation and ground-improvement assessment is recommended.")
        if liquefies else
        ("safe", "Prediction", "Liquefaction is unlikely",
         "These soil and loading conditions resemble cases that did not liquefy in the training data "
         "under the specified seismic demand.")
    )
    st.markdown(f"""
    <div class='verdict {css_class}'>
        <div class='label'>{label}</div>
        <div class='headline'>{headline}</div>
        <p>{text}</p>
        <div class='meter'>
            <div class='meter-track'>
                <div class='meter-mid'></div>
                <div class='meter-mark' style='left:{min(max(risk, 1), 99):.1f}%'></div>
            </div>
            <div class='meter-legend'><span>Stable</span><span>Threshold</span><span>Liquefies</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_stats:
    with st.container(border=True):
        st.metric("Risk score", f"{risk:.0f} / 100",
                  help="Decision score mapped to 0–100. 50 is the model's tuned decision threshold. "
                       "This is a relative index, not a calibrated probability.")
        st.metric("Margin from threshold", f"{decision - optimal_threshold:+.2f}",
                  help="Raw SVM decision score minus the tuned threshold. Positive → liquefaction.")
    if out_of_range:
        st.warning("Outside the range the model was trained on: **" + ", ".join(out_of_range)
                   + "**. Treat this result with extra caution.")

st.write("")

# --- DETAILS ---
tab_why, tab_whatif, tab_about = st.tabs(["Why this result", "What-if analysis", "About the model"])

with tab_why:
    st.markdown("How much each input pushes the prediction toward **liquefaction** or **stability**, "
                "compared with an average soil from the training data (SHAP values).")
    try:
        with st.spinner("Computing explanation…"):
            shap_vals, base_value = explain(tuple(float(user_input[c]) for c in feature_cols))
        contrib = pd.DataFrame({
            "feature": [f"{FEATURES[c]['label']} = {user_input[c]:g}" for c in feature_cols],
            "value": shap_vals,
        })
        contrib["direction"] = np.where(contrib["value"] >= 0, "Toward liquefaction", "Toward stability")
        contrib = contrib.reindex(contrib["value"].abs().sort_values(ascending=False).index)

        chart = (
            alt.Chart(contrib)
            .mark_bar(cornerRadiusEnd=4, height=26)
            .encode(
                x=alt.X("value:Q", title="Contribution to decision score"),
                y=alt.Y("feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
                color=alt.Color("direction:N",
                                scale=alt.Scale(domain=["Toward liquefaction", "Toward stability"],
                                                range=[RISK_COLOR, SAFE_COLOR]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=[alt.Tooltip("feature:N", title="Input"),
                         alt.Tooltip("value:Q", title="Contribution", format="+.3f")],
            )
            .properties(height=60 * len(contrib) + 20)
        )
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=MUTED).encode(x="x:Q")
        st.altair_chart(chart + zero, width="stretch")
        top = contrib.iloc[0]
        st.caption(
            f"Average-soil score {base_value:+.2f} + contributions = this soil's score {decision:+.2f} "
            f"(threshold {optimal_threshold:+.2f}). Biggest driver: **{top['feature']}** "
            f"({top['direction'].lower()})."
        )
    except Exception as e:
        st.warning(f"Could not generate the SHAP explanation: {e}")

with tab_whatif:
    st.markdown("Vary one parameter while keeping the others at their current values.")
    sweep_key = st.selectbox(
        "Parameter to vary", feature_cols, format_func=lambda c: FEATURES[c]["label"],
    )
    lo, hi = typical_range[sweep_key]
    lo, hi = min(lo, user_input[sweep_key]), max(hi, user_input[sweep_key])
    xs = np.linspace(lo, hi, 120)
    sweep = pd.DataFrame([user_input] * len(xs))
    sweep[sweep_key] = xs
    sweep["risk"] = risk_score(decision_scores(sweep))

    x_title = FEATURES[sweep_key]["label"] + (f" ({FEATURES[sweep_key]['unit']})" if FEATURES[sweep_key]["unit"] else "")
    base = alt.Chart(sweep).encode(x=alt.X(f"{sweep_key}:Q", title=x_title))
    line = base.mark_line(color="#2B2622", strokeWidth=2.5).encode(
        y=alt.Y("risk:Q", title="Risk score", scale=alt.Scale(domain=[0, 100])),
        tooltip=[alt.Tooltip(f"{sweep_key}:Q", title=FEATURES[sweep_key]["label"], format=".3g"),
                 alt.Tooltip("risk:Q", title="Risk score", format=".0f")],
    )
    bands = alt.Chart(pd.DataFrame({"y0": [0, 50], "y1": [50, 100], "c": [SAFE_COLOR, RISK_COLOR]})).mark_rect(
        opacity=0.08).encode(y="y0:Q", y2="y1:Q", color=alt.Color("c:N", scale=None))
    rule = alt.Chart(pd.DataFrame({"y": [50]})).mark_rule(strokeDash=[5, 4], color=MUTED).encode(y="y:Q")
    point = alt.Chart(pd.DataFrame({"x": [user_input[sweep_key]], "y": [risk]})).mark_point(
        size=140, filled=True, color=RISK_COLOR if liquefies else SAFE_COLOR, stroke="white", strokeWidth=2
    ).encode(x="x:Q", y="y:Q")
    st.altair_chart((bands + rule + line + point).properties(height=360), width="stretch")
    st.caption("Shaded region above the dashed line = predicted liquefaction. The dot marks your current input.")

with tab_about:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f"""
- **Model:** Support Vector Machine (`{type(model).__name__}`, {model.kernel.upper()} kernel, balanced class weights)
- **Inputs:** {len(feature_cols)} soil and loading parameters, standardised before prediction
- **Decision threshold:** {optimal_threshold:+.3f} on the SVM decision score (tuned during training)
- **Training samples:** {int(scaler.n_samples_seen_)}
        """)
    with c2:
        st.markdown("**Your input**")
        summary = pd.DataFrame({
            "Parameter": [FEATURES[c]["label"] for c in feature_cols],
            "Value": [f"{user_input[c]:g}" for c in feature_cols],
            "Unit": [FEATURES[c]["unit"] or "–" for c in feature_cols],
            "Training range": [f"{typical_range[c][0]:.3g} – {typical_range[c][1]:.3g}" for c in feature_cols],
        })
        st.dataframe(summary, hide_index=True, width="stretch")

st.markdown(
    "<div class='footer'>Decision-support tool for academic use. It does not replace a site-specific "
    "geotechnical assessment.</div>",
    unsafe_allow_html=True,
)
