
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("Gradient Boosting_model.pkl")

st.markdown("""
    <style>
    .main {background-color: #f0f8ff;}
    h1 {color: #2e8b57;}
    .stButton button {
        background-color: #008cba;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    .stDownloadButton button {
        background-color: #e67300;
        color: white;
        border-radius: 8px;
    }
    .stMarkdown {color: #4b0082;}
    </style>
""", unsafe_allow_html=True)

# App Config
st.set_page_config(page_title="💹 ESC 💲 Salary Classifier", page_icon="💹", layout="centered")

# Title and Tagline
st.title("Employee Salary Classification App")
st.markdown("""
This dazzling app predicts whether an employee earns **>50K or <=50K** annually.
Input data manually or try batch prediction with a CSV — now in living color! 🌈
""")

# Sidebar Input Form
st.sidebar.header("🎨 Enter Details with Flair")
with st.sidebar.expander("🌟 Individual Prediction", expanded=True):
    age = st.slider("🎂 Age", 18, 65, 30)
    education = st.selectbox("📚 Education", ["Bachelors", "Masters", "Phd", "HS-grad", "Assoc", "some-college"])
    occupation = st.selectbox("🧑‍🔧 Occupation", [
        "Tech-support", "Craft-repair", "Sales", "Exec-managerial", "Prof-speciality",
        "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
    ])
    hours_per_week = st.slider("🕓 Weekly Hours", 1, 80, 40)
    experience = st.slider("📈 Experience (Years)", 0, 40, 5)

input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

# Show input data
st.subheader("📋 Your Inputs")
st.dataframe(input_df.style.highlight_max(axis=0, color='lightgreen'))

# Prediction Button
if st.button("🚀 Predict Now", type="primary"):
    prediction = model.predict(input_df)
    st.success(f"🧠 AI thinks: **{prediction[0]}**")

# Batch Prediction Section
st.markdown("---")
st.subheader("📂 CSV Batch Prediction")

uploaded_file = st.file_uploader("📁 Upload CSV for Bulk Predictions", type=["csv"])
if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    st.write("🕵️ Preview Uploaded Data")
    st.dataframe(batch_data.style.highlight_max(axis=0, color='lightblue'))

    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = batch_preds

    st.write("🎯 Batch Predictions")
    st.dataframe(batch_data.style.highlight_max(axis=0, color='lightcoral'))

    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, file_name="Predicted_Results.csv", mime="text/csv")

st.markdown("💬 Made with 💖 by Tanishq")
