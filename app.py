import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Student Performance Prediction", layout="centered")

# Title
st.title("🎓 Student Performance Prediction App")

st.markdown("This app predicts whether a student will **Pass or Fail** using 3 ML models.")

# ===== Load trained models =====
@st.cache_resource
def load_models():
    lr = joblib.load("Logistic_Regression.pkl")
    rf = joblib.load("Random_Forest.pkl")
    svm = joblib.load("SVM.pkl")
    return lr, rf, svm

lr_model, rf_model, svm_model = load_models()

# ===== Sidebar Inputs =====
st.sidebar.header("📥 Enter Student Details")

study_hours = st.sidebar.number_input("Study Hours", min_value=0.0, max_value=24.0, value=2.0)
attendance = st.sidebar.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=75.0)
previous_score = st.sidebar.number_input("Previous Score", min_value=0.0, max_value=100.0, value=60.0)

# Input DataFrame
input_df = pd.DataFrame(
    [[study_hours, attendance, previous_score]],
    columns=["StudyHours", "Attendance", "PreviousScore"]
)

st.subheader("📊 Input Data")
st.dataframe(input_df)

# ===== Prediction =====
if st.button("🔮 Predict Result"):
    lr_pred = lr_model.predict(input_df)[0]
    rf_pred = rf_model.predict(input_df)[0]
    svm_pred = svm_model.predict(input_df)[0]

    def result_text(pred):
        return "✅ Pass" if pred == 1 else "❌ Fail"

    st.subheader("🧠 Model Predictions")
    st.write("Logistic Regression:", result_text(lr_pred))
    st.write("Random Forest:", result_text(rf_pred))
    st.write("SVM:", result_text(svm_pred))

# ===== F1 Score Display (Hardcoded from training) =====
st.subheader("📌 F1 Score Comparison")

model_names = ["Logistic Regression", "Random Forest", "SVM"]
f1_scores = [0.78, 0.85, 0.80]   # Replace with your real F1 scores

f1_df = pd.DataFrame({
    "Model": model_names,
    "F1 Score": f1_scores
})

st.table(f1_df)

# ===== Visualization 1: F1 Score Bar Chart =====
st.subheader("📈 Visualization 1: F1 Score Bar Chart")
fig1, ax1 = plt.subplots()
ax1.bar(model_names, f1_scores)
ax1.set_ylabel("F1 Score")
ax1.set_title("Model Performance (F1 Score)")
st.pyplot(fig1)

# ===== Visualization 2: Prediction Comparison =====
st.subheader("📊 Visualization 2: Prediction Output (Dummy Example)")
pred_values = [lr_pred if 'lr_pred' in locals() else 0,
               rf_pred if 'rf_pred' in locals() else 0,
               svm_pred if 'svm_pred' in locals() else 0]

fig2, ax2 = plt.subplots()
ax2.bar(model_names, pred_values)
ax2.set_ylabel("Prediction (0 = Fail, 1 = Pass)")
ax2.set_title("Model Predictions Comparison")
st.pyplot(fig2)

# ===== Visualization 3: Input Feature Chart =====
st.subheader("📉 Visualization 3: Input Feature Values")

fig3, ax3 = plt.subplots()
ax3.bar(["StudyHours", "Attendance", "PreviousScore"],
        [study_hours, attendance, previous_score])
ax3.set_title("Student Input Features")
st.pyplot(fig3)

st.markdown("---")
st.caption("🚀 Built using Streamlit + Machine Learning")
