"""Public-facing diabetes screening questionnaire and result page."""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import core
import utils

utils.setup_page("Screening Assistant")
meta = utils.load_meta()
pipe = utils.load_pipeline()
features = meta["features"]
threshold, overridden = utils.active_threshold(meta)
bmi_min = float(meta["input_ranges"]["BMI"]["min"])
bmi_max = float(meta["input_ranges"]["BMI"]["max"])

st.markdown('<div class="ds-eyebrow">PERSONAL SCREENING SUPPORT</div>', unsafe_allow_html=True)
st.title("Understand your diabetes risk factors")
st.markdown(
    '<p class="ds-subtitle">Answer seven simple questions to see how your responses compare with patterns '
    'in a US health survey. This educational tool can help you decide whether to discuss a blood glucose '
    'test with a healthcare professional.</p>',
    unsafe_allow_html=True,
)
st.info(
    "Important: this tool does not diagnose diabetes and does not calculate your true personal probability. "
    "It uses a model trained on self-reported data from US adults surveyed in 2015. It has not been validated "
    "for clinical use or for the Vietnamese population."
)

with st.expander("Before you begin", expanded=False):
    st.markdown(
        "- This questionnaire is intended for adults aged 18 or older.\n"
        "- Answer based on what a healthcare professional has told you and how you generally feel.\n"
        "- Your answers are used to calculate a result in this app session; this project does not save them to a database.\n"
        "- If you feel unwell or have symptoms that concern you, seek medical advice regardless of the result."
    )

with st.form("screening_form", border=True):
    st.subheader("1. About you")
    c1, c2 = st.columns(2)
    with c1:
        age_key = st.selectbox(
            "What is your age group?",
            options=list(core.AGE_BANDS),
            index=None,
            placeholder="Select your age group",
            format_func=lambda k: core.AGE_BANDS[k],
        )
        height = st.number_input("Height (cm)", min_value=120, max_value=220, value=165, step=1,
                                 help="Enter a value between 120 and 220 cm.")
    with c2:
        weight = st.number_input("Weight (kg)", min_value=30, max_value=250, value=65, step=1,
                                 help="Enter a value between 30 and 250 kg.")
        genhlth = st.selectbox(
            "How would you rate your health in general?",
            options=list(core.GENHLTH_LABELS),
            index=None,
            placeholder="Choose one answer",
            format_func=lambda k: core.GENHLTH_LABELS[k],
        )

    st.divider()
    st.subheader("2. Health history")
    st.caption("For each question, select Yes or No. Choose based on what a healthcare professional has told you.")
    c1, c2 = st.columns(2)
    with c1:
        highbp = st.radio(
            "Have you ever been told you have high blood pressure?",
            ["Yes", "No"], index=None, horizontal=True, key="screen_highbp",
        )
        highchol = st.radio(
            "Have you ever been told you have high cholesterol?",
            ["Yes", "No"], index=None, horizontal=True, key="screen_highchol",
        )
    with c2:
        heart = st.radio(
            "Have you ever been told you have coronary heart disease or had a heart attack?",
            ["Yes", "No"], index=None, horizontal=True, key="screen_heart",
        )
        diffwalk = st.radio(
            "Do you have serious difficulty walking or climbing stairs?",
            ["Yes", "No"], index=None, horizontal=True, key="screen_diffwalk",
        )

    submitted = st.form_submit_button("Review my answers", type="primary", use_container_width=True)

if submitted:
    missing = []
    for label, value in (
        ("age group", age_key), ("general health", genhlth),
        ("high blood pressure", highbp), ("high cholesterol", highchol),
        ("heart disease / heart attack", heart), ("difficulty walking", diffwalk),
    ):
        if value is None:
            missing.append(label)
    if missing:
        st.session_state.pop("screening_result", None)
        st.error("Please answer every question before continuing. Missing: " + ", ".join(missing) + ".")
    else:
        bmi = float(weight / (height / 100) ** 2)
        if not bmi_min <= bmi <= bmi_max:
            st.session_state.pop("screening_result", None)
            st.error(
                f"The BMI calculated from those measurements is {bmi:.1f}. "
                f"This model accepts BMI values from {bmi_min:g} to {bmi_max:g}. "
                "Please check the height and weight you entered. Your value has not been adjusted automatically."
            )
        else:
            row = {
                "Age": float(age_key),
                "BMI": bmi,
                "HighBP": float(highbp == "Yes"),
                "HighChol": float(highchol == "Yes"),
                "GenHlth": float(genhlth),
                "DiffWalk": float(diffwalk == "Yes"),
                "HeartDiseaseorAttack": float(heart == "Yes"),
            }
            x = pd.DataFrame([row])[features]
            score = float(pipe.predict_proba(x)[0, 1])
            contrib = core.local_contributions(pipe, x, features, meta["reference_values"])
            st.session_state["screening_result"] = {
                "x": x, "score": score, "contrib": contrib,
                "answers": {
                    "age": core.AGE_BANDS[age_key],
                    "height": int(height), "weight": int(weight),
                    "general_health": core.GENHLTH_LABELS[genhlth],
                    "highbp": highbp, "highchol": highchol,
                    "heart": heart, "diffwalk": diffwalk,
                },
            }

res = st.session_state.get("screening_result")
if res is None:
    st.markdown(
        '<div class="ds-panel"><strong>What happens next?</strong><br>'
        'Complete the questions above. We will show a model score, explain the threshold, '
        'and provide a plain-language suggestion for your next step.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

score, x, contrib = res["score"], res["x"], res["contrib"]
flagged = score >= threshold
base_rate = float(meta["data"]["prevalence_used"])
bmi = float(x.iloc[0]["BMI"])

st.divider()
st.markdown('<div class="ds-eyebrow">YOUR SCREENING SUMMARY</div>', unsafe_allow_html=True)
st.header("Your result")
m1, m2, m3 = st.columns(3)
m1.metric(
    "Model score", f"{score:.1%}",
    help="A model output based on patterns in the survey. It is not your personal probability of having diabetes.",
)
m2.metric(
    "Decision threshold", f"{threshold:.1%}" + (" · adjusted" if overridden else ""),
    help="The score level used to flag a response. The default comes from the project's validation analysis.",
)
m3.metric("Survey positive-label rate", f"{base_rate:.1%}",
          help="Share of records in the cleaned survey that reported diabetes or prediabetes.")
st.caption(f"Calculated BMI: **{bmi:.1f}** · The model uses your BMI as one of seven inputs.")

if flagged:
    st.warning(
        "**Your responses were flagged by the model.** Consider speaking with a qualified healthcare "
        "professional about whether a blood glucose test, such as HbA1c or fasting blood glucose, is appropriate."
    )
else:
    st.success(
        "**Your responses were not flagged at the current threshold.** This does not rule out diabetes or "
        "prediabetes. Continue routine care and speak with a healthcare professional if you have symptoms, "
        "risk concerns, or questions."
    )

st.markdown(
    '<div class="ds-note"><strong>Please remember:</strong> a flag is not a diagnosis, and a result below '
    'the threshold is not a guarantee of good health. Only appropriate clinical assessment and blood tests '
    'can diagnose diabetes or prediabetes.</div>',
    unsafe_allow_html=True,
)

st.subheader("What influenced the model score?")
st.caption(
    "These are approximate model-behaviour explanations. Each value compares your answer with a typical "
    "answer in the training data; it does not show cause and effect."
)
shown = contrib[contrib["effect"].abs() >= 0.001].copy()
shown["abs_effect"] = shown["effect"].abs()
shown = shown.sort_values("abs_effect", ascending=False).head(5).iloc[::-1]

if shown.empty:
    st.info("No single answer moved the model score noticeably compared with the typical reference answers.")
else:
    fig, ax = plt.subplots(figsize=(8, max(2.6, 0.55 * len(shown) + 1)))
    labels = [
        f"{meta['feature_labels'][f]} · {core.format_value(f, v)}"
        for f, v in zip(shown["feature"], shown["your_value"])
    ]
    colors = ["#16877d" if e > 0 else "#7b9eaa" for e in shown["effect"]]
    ax.barh(labels, shown["effect"] * 100, color=colors, height=0.62)
    ax.axvline(0, color="#607674", lw=0.9)
    ax.set_xlabel("Difference in model score (percentage points)")
    ax.set_title("Approximate change when one answer is replaced by a typical answer")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", alpha=0.18)
    st.pyplot(fig, clear_figure=True)
    st.caption("Positive values mean your answer raises the model score relative to the typical reference answer; negative values mean it lowers the score.")

answer_table = pd.DataFrame([
    {"Question": "Age group", "Your answer": res["answers"]["age"]},
    {"Question": "Height", "Your answer": f"{res['answers']['height']} cm"},
    {"Question": "Weight", "Your answer": f"{res['answers']['weight']} kg"},
    {"Question": "Body mass index (BMI)", "Your answer": f"{bmi:.1f}"},
    {"Question": "General health", "Your answer": res["answers"]["general_health"]},
    {"Question": "High blood pressure", "Your answer": res["answers"]["highbp"]},
    {"Question": "High cholesterol", "Your answer": res["answers"]["highchol"]},
    {"Question": "Heart disease or heart attack", "Your answer": res["answers"]["heart"]},
    {"Question": "Serious difficulty walking", "Your answer": res["answers"]["diffwalk"]},
])
with st.expander("Review the answers you submitted"):
    st.dataframe(answer_table, hide_index=True, use_container_width=True)
    detail = contrib.copy()
    detail["Question"] = detail["feature"].map(meta["feature_labels"])
    detail["Your answer"] = [core.format_value(f, v) for f, v in zip(detail["feature"], detail["your_value"])]
    detail["Typical survey answer"] = [core.format_value(f, v) for f, v in zip(detail["feature"], detail["typical_value"])]
    detail["Score difference (pp)"] = (detail["effect"] * 100).round(2)
    st.dataframe(detail[["Question", "Your answer", "Typical survey answer", "Score difference (pp)"]],
                 hide_index=True, use_container_width=True)

if st.button("Start a new screening", type="secondary"):
    st.session_state.pop("screening_result", None)
    for key in ("screen_highbp", "screen_highchol", "screen_heart", "screen_diffwalk"):
        st.session_state.pop(key, None)
    st.rerun()
