# learner_app.py - Streamlit Dashboard with Hugging Face Suggestions (No OpenAI required)

import streamlit as st
import pandas as pd
import joblib
import os
import requests
import warnings

# --- Page Configuration ---
st.set_page_config(
    page_title="Slow Learner Prediction Tool",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Artifacts ---
@st.cache_resource
def load_artifacts():
    try:
        scaler = joblib.load("scaler.joblib")
        model = joblib.load("random_forest_model.joblib")
        features = joblib.load("feature_names.joblib")
        return scaler, model, features
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

scaler, model, feature_names = load_artifacts()

# --- Input Configuration ---
main_inputs = [
    'study_hours_per_day', 'social_media_hours', 'attendance_percentage',
    'sleep_hours', 'exam_score', 'mental_health_rating'
]

categorical_inputs = {
    'gender': ['Female', 'Male', 'Other'],
    'part_time_job': ['No', 'Yes'],
    'diet_quality': ['Average', 'Good', 'Poor'],
    'parental_education_level': ['Bachelor', 'High School', 'Master', 'Unknown'],
    'internet_quality': ['Average', 'Good', 'Poor'],
    'extracurricular_participation': ['No', 'Yes']
}

# --- Helper Functions ---
def prepare_input(user_input, cat_input, expected_features):
    df = pd.DataFrame([user_input])
    for col, value in cat_input.items():
        df[col] = value
    df = pd.get_dummies(df, drop_first=True)
    for feat in expected_features:
        if feat not in df.columns:
            df[feat] = 0
    df = df[expected_features]
    return df

def predict_support(df, scaler, model):
    scaled = scaler.transform(df)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]
    return pred, prob

def calculate_risk_score(study_hours, exam_score, attendance, participation, sleep_hours, social_media, prob):
    score = 0
    if study_hours < 2:
        score += 25
    if exam_score < 65:
        score += 25
    if attendance < 85:
        score += 10
    if participation <= 2:
        score += 10
    if sleep_hours < 6:
        score += 10
    if social_media > 4:
        score += 10
    if prob > 0.35:
        score += 10
    return score

def get_remedial_suggestions(score, probability, exam_score,
                              attendance_percentage, participation_rating,
                              study_hours, sleep_hours, social_media, part_time, extracurricular,
                              mental_health_rating):
    suggestions = []
    avg_metric = (study_hours * 10 + sleep_hours * 10 + mental_health_rating + exam_score + attendance_percentage + (6 - participation_rating) * 10 + (6 - mental_health_rating)) / 7
    threshold = 60

    if score >= 50 or avg_metric < threshold:
        suggestions.append(f"**Student may benefit from additional support (Risk Score: {score}/100)**")
        suggestions.append("---")
        suggestions.append("### ✅ General Support Recommendations")
        suggestions.extend([
            "* Meet individually to identify challenges and learning preferences.",
            "* Break down complex topics with step-by-step guidance.",
            "* Provide more practice in weaker subjects.",
            "* Incorporate visual learning aids and activities.",
            "* Encourage active participation in a supportive setting.",
            "* Recommend mentoring or peer learning sessions."
        ])

        suggestions.append("\n---")
        suggestions.append("### 📌 Personalized Observations & Strategies")
        triggered_specific = False

        if exam_score < 45:
            suggestions.append("* *Observation:* Very Low Standardized Test Score → Focus on core concepts and regular practice.")
            triggered_specific = True

        if attendance_percentage < 75:
            suggestions.append("* *Observation:* Low Attendance → Discuss reasons and promote regular class routines.")
            triggered_specific = True

        if participation_rating <= 3:
            suggestions.append("* *Observation:* Low Class Participation → Offer incentives and create non-judgmental space.")
            triggered_specific = True

        if study_hours < 2:
            suggestions.append("* *Observation:* Insufficient Study Time → Design time tables with short, focused sessions.")
            triggered_specific = True

        if sleep_hours < 6:
            suggestions.append("* *Observation:* Poor Sleep Habits → Promote healthy sleep routines (7-8 hrs/night).")
            triggered_specific = True

        if social_media > 4:
            suggestions.append("* *Observation:* High Screen Time → Introduce productivity tools and digital detox plans.")
            triggered_specific = True

        if part_time == "Yes" and study_hours < 2:
            suggestions.append("* *Observation:* Work-Study Conflict → Balance workload and offer weekend sessions.")
            triggered_specific = True

        if extracurricular == "Yes" and exam_score < 60:
            suggestions.append("* *Observation:* Over-scheduled with Activities → Temporarily reduce extracurricular load.")
            triggered_specific = True

        if not triggered_specific:
            suggestions.append("* No individual issues flagged. Support focus, motivation, and consistency.")

        suggestions.append("\n---")
        suggestions.append("**🔁 Follow-Up:** Review student progress weekly and adjust interventions as needed.")

    return suggestions

def get_hf_suggestions(student_profile_dict):
    prompt = f"""
    A student is showing signs of struggling in academics. Their profile is:
    {student_profile_dict}

    Based on this, generate 3 short, practical, and personalized remedial strategies.
    """
    headers = {
        "Authorization": f"Bearer {st.secrets['HF_API_KEY']}"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.7,
            "max_length": 250
        }
    }
    model_url = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
    try:
        response = requests.post(model_url, headers=headers, json=payload)
        result = response.json()
        if isinstance(result, dict) and "error" in result:
            return f"❌  API Error: {result['error']}"
        return result[0]['generated_text']
    except Exception as e:
        return f"⚠️ Error getting  suggestions: {e}"

# --- UI ---
st.title("🎓 Slow Learner Prediction Tool")
st.subheader("Identify students who might need additional support")

with st.sidebar.form("student_form"):
    st.markdown("### 🎯 Core Student Features")

    study_hours = st.number_input("Study Hours Per Day", 0.0, 24.0, 2.0, 0.5)
    social_media = st.number_input("Social Media Hours", 0.0, 24.0, 1.0, 0.5)
    attendance = st.number_input("Attendance Percentage", 0.0, 100.0, 90.0, 0.5)
    sleep_hours = st.number_input("Sleep Hours Per Day", 0.0, 24.0, 7.0, 0.5)
    exam_score = st.number_input("Exam Score", 0.0, 100.0, 60.0, 0.5)
    mental_health_rating = st.slider("Mental Health Rating", 1, 5, 3)
    participation = st.slider("Participation Rating", 1, 5, 3)

    with st.expander("➕ Optional Features (Category Inputs)"):
        gender = st.selectbox("Gender", categorical_inputs['gender'])
        part_time = st.selectbox("Part-Time Job", categorical_inputs['part_time_job'])
        diet = st.selectbox("Diet Quality", categorical_inputs['diet_quality'])
        parent_edu = st.selectbox("Parental Education Level", categorical_inputs['parental_education_level'])
        net_quality = st.selectbox("Internet Quality", categorical_inputs['internet_quality'])
        extracurricular = st.selectbox("Extracurricular Participation", categorical_inputs['extracurricular_participation'])

    use_ai = st.checkbox("💬 Use AI Suggestions ", value=True)
    submit_btn = st.form_submit_button("✨ Predict Support Need")

# --- Main Panel ---
if submit_btn:
    if scaler is None or model is None or feature_names is None:
        st.error("Artifacts not loaded. Please check the model files.")
    else:
        with st.spinner("Analyzing student data..."):
            user_inputs = {
                'study_hours_per_day': study_hours,
                'social_media_hours': social_media,
                'attendance_percentage': attendance,
                'sleep_hours': sleep_hours,
                'exam_score': exam_score,
                'mental_health_rating': mental_health_rating
            }
            cat_values = {
                'gender': gender,
                'part_time_job': part_time,
                'diet_quality': diet,
                'parental_education_level': parent_edu,
                'internet_quality': net_quality,
                'extracurricular_participation': extracurricular
            }
            input_df = prepare_input(user_inputs, cat_values, feature_names)
            prediction, probability = predict_support(input_df, scaler, model)
            risk_score = calculate_risk_score(study_hours, exam_score, attendance, participation, sleep_hours, social_media, probability)

        if risk_score >= 30:
            st.error(f"🚨 This student may be a slow learner (Risk Score: {risk_score}/100)")
        else:
            st.success("✅ No significant learning difficulties detected.")

        st.progress(risk_score / 100.0, text=f"Slow Learner Risk Score: {risk_score}/100")

        suggestions = get_remedial_suggestions(
            risk_score, probability, exam_score, attendance, participation,
            study_hours, sleep_hours, social_media, part_time, extracurricular, mental_health_rating
        )

        if suggestions:
            with st.expander("💡 Suggested Remedial Actions", expanded=True):
                for tip in suggestions:
                    st.markdown(tip)

        if use_ai:
            st.markdown("### 🤖 AI-Based Suggestions (Hugging Face)")
            student_data = {
                "Study Hours": study_hours,
                "Exam Score": exam_score,
                "Attendance": attendance,
                "Participation": participation,
                "Sleep Hours": sleep_hours,
                "Social Media Hours": social_media,
                "Part-time Job": part_time,
                "Extracurricular Activities": extracurricular,
                "Mental Health Rating": mental_health_rating
            }
            hf_advice = get_hf_suggestions(student_data)
            st.markdown(hf_advice)

st.markdown("---")
st.caption("Disclaimer: This is an AI-based tool. Use results alongside academic judgment.")
