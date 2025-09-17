# app.py
import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

# -----------------------------
# Load dataset and model
# -----------------------------
df = pd.read_csv("StudentsPerformance.csv")
model = joblib.load("student_performance_model.pkl")
model_columns = joblib.load("model_columns.pkl")  # saved during training

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filter Students")
gender_filter = st.sidebar.multiselect("Gender", df["gender"].unique(), default=df["gender"].unique())
race_filter = st.sidebar.multiselect("Race/Ethnicity", df["race/ethnicity"].unique(), default=df["race/ethnicity"].unique())
prep_filter = st.sidebar.multiselect("Test Preparation Course", df["test preparation course"].unique(),
                                     default=df["test preparation course"].unique())

filtered_df = df[(df["gender"].isin(gender_filter)) &
                 (df["race/ethnicity"].isin(race_filter)) &
                 (df["test preparation course"].isin(prep_filter))]

st.title("Student Performance Dashboard")
st.dataframe(filtered_df)

# -----------------------------
# Prediction Form
# -----------------------------
st.header("Predict Total Score")
with st.form("prediction_form"):
    gender = st.selectbox("Gender", df["gender"].unique())
    race = st.selectbox("Race/Ethnicity", df["race/ethnicity"].unique())
    prep = st.selectbox("Test Preparation Course", df["test preparation course"].unique())
    math_score = st.number_input("Math Score", 0, 100, 50)
    reading_score = st.number_input("Reading Score", 0, 100, 50)
    writing_score = st.number_input("Writing Score", 0, 100, 50)
    submitted = st.form_submit_button("Predict")

if submitted:
    # Prepare input data
    input_dict = {
        "gender": [gender],
        "race/ethnicity": [race],
        "test preparation course": [prep],
        "math score": [math_score],
        "reading score": [reading_score],
        "writing score": [writing_score]
    }
    input_df = pd.DataFrame(input_dict)
    
    # Align with model columns
    input_df = pd.get_dummies(input_df)
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_columns]
    
    predicted_total = model.predict(input_df)[0]
    st.success(f"Predicted Total Score: {predicted_total:.2f}")

# -----------------------------
# Visualizations
# -----------------------------
st.header("Visualizations")

# Grade column
filtered_df["total score"] = filtered_df["math score"] + filtered_df["reading score"] + filtered_df["writing score"]
bins = [0, 180, 240, 300]
labels = ["C", "B", "A"]
filtered_df["grade"] = pd.cut(filtered_df["total score"], bins=bins, labels=labels, include_lowest=True)

# Seaborn countplot for grades
fig, ax = plt.subplots()
sns.countplot(x="grade", data=filtered_df, palette="coolwarm", ax=ax)
ax.set_title("Grade Distribution")
st.pyplot(fig)

# -----------------------------
# Average Total Score by Test Preparation Course
# -----------------------------
prep_chart = alt.Chart(filtered_df).mark_bar().encode(
    x="test preparation course",
    y="mean(total score)",
    color="test preparation course",
    tooltip=["test preparation course", "mean(total score)"]
).properties(title="Average Total Score by Test Preparation Course")
st.altair_chart(prep_chart, use_container_width=True)

# -----------------------------
# Average Total Score by Parental Level of Education
# -----------------------------
parent_chart = alt.Chart(filtered_df).mark_bar().encode(
    x="parental level of education",
    y="mean(total score)",
    color="parental level of education",
    tooltip=["parental level of education", "mean(total score)"]
).properties(title="Average Total Score by Parental Level of Education")
st.altair_chart(parent_chart, use_container_width=True)

# -----------------------------
# Score Correlation Heatmap
# -----------------------------
score_corr = filtered_df[["math score", "reading score", "writing score"]].corr().reset_index().melt('index')
heatmap = alt.Chart(score_corr).mark_rect().encode(
    x='index',
    y='variable',
    color='value',
    tooltip=['index', 'variable', 'value']
).properties(title="Correlation Heatmap of Scores")
st.altair_chart(heatmap, use_container_width=True)

# -----------------------------
# CSV Download
# -----------------------------
st.download_button(
    label="Download Filtered Data as CSV",
    data=filtered_df.to_csv(index=False),
    file_name='filtered_students.csv',
    mime='text/csv'
)
