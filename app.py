"""Public overview dashboard: explain the project and its evidence."""
import matplotlib.pyplot as plt
import streamlit as st

import utils

utils.setup_page("Overview", "🩺")
meta = utils.load_meta()
data = meta["data"]
metrics = meta["test_metrics"]
at = metrics["at_threshold"]

st.markdown('<div class="ds-eyebrow">WELCOME</div>', unsafe_allow_html=True)
st.title("A clearer starting point for diabetes awareness")
st.markdown(
    '<p class="ds-subtitle">Explore health indicators, try a short educational screening questionnaire, '
    'and learn how a machine-learning model behaves. This project is designed to support awareness — '
    'not to replace a healthcare professional.</p>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.35, 1])
with left:
    st.markdown("### What you can do here")
    st.markdown(
        "- **Screening Assistant:** answer seven questions and review an educational model result.\n"
        "- **Guided Health Assistant:** learn about diabetes, screening tests, BMI, and how to interpret the result.\n"
        "- **Model & fairness pages:** explore test-set performance, threshold trade-offs, and group-level evaluation."
    )
    st.markdown(
        '<div class="ds-note"><strong>Not a medical service.</strong> The model was trained on a 2015 US survey '
        'and has not been validated for clinical decisions or for people in Vietnam.</div>',
        unsafe_allow_html=True,
    )
with right:
    st.markdown("### At a glance")
    st.metric("Survey responses analyzed", f"{data['n_used']:,}")
    st.metric("Reported diabetes / prediabetes", f"{data['prevalence_used']:.1%}")
    st.metric("Questions used by the app model", f"{len(meta['features'])}")

st.divider()
st.subheader("How the project works")
steps = st.columns(3)
steps[0].markdown("#### 1 · Answer")
steps[0].write("Provide a small set of health-related answers. No lab result is required.")
steps[1].markdown("#### 2 · Compare")
steps[1].write("A saved machine-learning pipeline compares the response pattern with patterns learned from the survey.")
steps[2].markdown("#### 3 · Learn")
steps[2].write("Review the model output and consider whether to discuss testing with a qualified professional.")

st.divider()
st.subheader("How well did the model perform?")
st.caption("These figures come from a held-out test set. They describe model performance on survey records, not your personal health.")
m = st.columns(4)
m[0].metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
m[1].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
m[2].metric("Recall", f"{at['recall']:.1%}")
m[3].metric("Precision", f"{at['precision']:.1%}")
st.caption(
    f"At the project's default threshold of {meta['threshold']:.2f}, recall is the share of positive survey-label "
    f"cases flagged by the model; precision is the share of flagged records that have the positive survey label."
)

st.subheader("What do recall and precision mean?")
a, b = st.columns(2)
with a:
    st.markdown("**Recall — how many labelled cases are caught**")
    st.write(f"The model flags about {at['recall']:.0%} of people in the test set who reported diabetes or prediabetes.")
with b:
    st.markdown("**Precision — how often a flag matches the survey label**")
    st.write(f"About {at['precision']:.0%} of records flagged by the model reported diabetes or prediabetes.")

st.divider()
st.subheader("Where does the data come from?")
st.write(
    f"The project uses the US Centers for Disease Control and Prevention's Behavioral Risk Factor Surveillance "
    f"System (BRFSS) 2015 Diabetes Health Indicators dataset. It contains {data['n_raw']:,} original survey "
    f"responses; after removing {data['n_duplicate_rows']:,} exact duplicate rows, {data['n_used']:,} records "
    "were used."
)
st.warning(
    "The target is self-reported diabetes or prediabetes, not a laboratory-confirmed diagnosis. "
    "The survey is from the United States in 2015, so the results may not transfer to other countries, "
    "populations, or current clinical settings."
)
