"""Guided educational assistant with fixed, safety-reviewed topics."""
import streamlit as st

import utils

utils.setup_page("Guided Health Assistant", "💬")

st.markdown('<div class="ds-eyebrow">LEARN AT YOUR OWN PACE</div>', unsafe_allow_html=True)
st.title("Guided Health Assistant")
st.markdown(
    '<p class="ds-subtitle">Choose a topic to get a short, plain-language explanation and practical questions '
    'you may want to discuss with a healthcare professional.</p>',
    unsafe_allow_html=True,
)
st.info(
    "This is an educational guide, not a medical chatbot. It cannot assess symptoms, diagnose a condition, "
    "recommend treatment, or replace advice from a qualified healthcare professional."
)

topics = {
    "Diabetes and prediabetes": {
        "intro": "Diabetes is a condition in which blood glucose (sugar) stays too high. Prediabetes means blood glucose is higher than usual but not yet in the diabetes range. A blood test is needed to assess these conditions.",
        "points": [
            "Type 2 diabetes is commonly associated with insulin resistance and changes in how the body regulates blood glucose.",
            "Some people have no obvious symptoms, so screening may be discussed even when someone feels well.",
            "A screening questionnaire cannot confirm or rule out diabetes.",
        ],
        "ask": "Would a blood glucose test be appropriate for me, considering my age, health history, and risk factors?",
    },
    "Understanding the screening result": {
        "intro": "The score shown in this project is a machine-learning output. It reflects how closely the answers match patterns associated with self-reported diabetes or prediabetes in the training survey.",
        "points": [
            "It is not your personal probability of having diabetes.",
            "A result above the threshold means the model flags the response pattern; it does not mean you have diabetes.",
            "A result below the threshold does not rule out diabetes or mean that testing is unnecessary.",
            "The model was trained on US survey data from 2015 and has not been validated for clinical use in Vietnam.",
        ],
        "ask": "Given my circumstances, should I have a laboratory blood glucose test?",
    },
    "HbA1c and blood glucose tests": {
        "intro": "Healthcare professionals may use blood tests to assess blood glucose. Two commonly discussed tests are HbA1c and fasting blood glucose. Which test is suitable depends on the person's circumstances and local clinical guidance.",
        "points": [
            "HbA1c reflects average blood glucose over roughly the previous two to three months.",
            "A fasting blood glucose test measures blood glucose after a period without food, according to the laboratory's instructions.",
            "Test preparation, interpretation, and the need for repeat testing should be discussed with a healthcare professional.",
        ],
        "ask": "Which test is suitable for me, and do I need to prepare or fast beforehand?",
    },
    "Understanding BMI": {
        "intro": "Body mass index (BMI) is calculated from weight and height: weight in kilograms divided by height in metres squared. It is one broad measure, not a complete assessment of health.",
        "points": [
            "BMI does not directly measure blood glucose or diagnose diabetes.",
            "It does not describe body-fat distribution or every aspect of an individual's health.",
            "The model uses BMI as one of seven inputs; its contribution is a description of model behaviour, not proof of cause.",
        ],
        "ask": "How should my BMI be interpreted alongside my medical history and other health measures?",
    },
    "Healthy habits and prevention": {
        "intro": "Everyday habits can support general health. The right approach depends on an individual's health, ability, preferences, and access to care.",
        "points": [
            "Choose balanced meals that are realistic and sustainable for you.",
            "Regular movement that suits your ability can support general health.",
            "Sleep, stress, smoking, and routine healthcare are also relevant to overall wellbeing.",
            "Avoid extreme diets or making medication changes without professional advice.",
        ],
        "ask": "What small, realistic changes would be appropriate and safe for my health situation?",
    },
}

topic = st.selectbox("What would you like to learn about?", list(topics), index=0)
item = topics[topic]
st.divider()
st.subheader(topic)
st.write(item["intro"])
st.markdown("**Key points**")
for point in item["points"]:
    st.markdown(f"- {point}")
st.markdown("**A question you could ask a healthcare professional**")
st.markdown(f'<div class="ds-note">{item["ask"]}</div>', unsafe_allow_html=True)

with st.expander("When should I seek medical help?"):
    st.write(
        "If you have symptoms or concerns, contact a healthcare professional rather than relying on this app. "
        "Seek urgent medical care for severe or rapidly worsening symptoms, such as confusion, fainting, "
        "difficulty breathing, or severe weakness."
    )

st.caption(
    "Content is general education only. It is not a personalised medical recommendation and may not reflect "
    "the latest local clinical guidance."
)
