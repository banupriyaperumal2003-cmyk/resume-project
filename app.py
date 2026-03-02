import streamlit as st
import PyPDF2
import re
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📊",
    layout="wide"
)

# ---------------- SKILL DATABASE ----------------
SKILLS_DB = [
    "python", "java", "c++", "sql", "mysql", "mongodb",
    "react", "angular", "html", "css", "javascript",
    "spring", "spring boot", "django", "flask",
    "machine learning", "deep learning", "data analysis",
    "pandas", "numpy", "tensorflow", "scikit-learn",
    "aws", "docker", "kubernetes", "git"
]

# ---------------- FUNCTIONS ----------------
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text.lower()

def extract_skills(text):
    text = text.lower()
    found_skills = []
    for skill in SKILLS_DB:
        if skill in text:
            found_skills.append(skill)
    return sorted(found_skills)

# ---------------- HEADER ----------------
st.title("📊 AI-Based Resume Analyzer")
st.markdown("Compare your resume with a job description using AI & NLP techniques.")

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

with col2:
    job_description = st.text_area("Paste Job Description Here")

# ---------------- ANALYZE BUTTON ----------------
if st.button("Analyze Resume"):

    if uploaded_resume is None or job_description.strip() == "":
        st.warning("Please upload resume and enter job description.")
    else:

        resume_text = extract_text_from_pdf(uploaded_resume)
        job_text = job_description.lower()

        # -------- TEXT SIMILARITY --------
        documents = [resume_text, job_text]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)

        similarity_score = cosine_similarity(
            tfidf_matrix[0:1], tfidf_matrix[1:2]
        )[0][0]

        text_match_percentage = round(similarity_score * 100, 2)

        # -------- SKILL MATCHING --------
        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_text)

        matched_skills = sorted(list(set(resume_skills) & set(job_skills)))
        missing_skills = sorted(list(set(job_skills) - set(resume_skills)))

        skill_match_percentage = (
            (len(matched_skills) / len(job_skills)) * 100
            if job_skills else 0
        )

        # ---------------- RESULTS ----------------
        st.divider()
        st.subheader("📊 Skill Match Summary")

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Required Skills", len(job_skills))
        c2.metric("Matched Skills", len(matched_skills))
        c3.metric("Missing Skills", len(missing_skills))

        st.write("### 🧠 Skill Match Score")
        st.progress(int(skill_match_percentage))
        st.write(f"{round(skill_match_percentage,2)}%")

        st.write("### 📄 Text Similarity Score")
        st.progress(min(int(text_match_percentage), 100))
        st.write(f"{text_match_percentage}%")

        # -------- OVERALL FEEDBACK --------
        if skill_match_percentage >= 75:
            st.success("Excellent Skill Match ✅")
        elif skill_match_percentage >= 50:
            st.warning("Moderate Skill Match ⚠")
        else:
            st.error("Low Skill Match ❌")

        # ---------------- PIE CHART ----------------
        st.divider()
        st.subheader("📈 Skill Distribution")

        chart_data = pd.DataFrame({
            "Category": ["Matched", "Missing"],
            "Count": [len(matched_skills), len(missing_skills)]
        })

        fig = px.pie(
            chart_data,
            names="Category",
            values="Count",
            color="Category",
            color_discrete_map={
                "Matched": "#4B0082",   # Dark Purple
                "Missing": "#4E342E"    # Dark Brown
            }
        )

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font_color="black",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------- SKILL DETAILS ----------------
        st.divider()

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("🟣 Matched Skills")
            if matched_skills:
                for skill in matched_skills:
                    st.write(f"✔ {skill}")
            else:
                st.write("No matched skills found")

        with col_right:
            st.subheader("🟤 Missing Skills")
            if missing_skills:
                for skill in missing_skills:
                    st.write(f"✘ {skill}")
            else:
                st.write("No missing skills 🎉")

        # ---------------- AI SUGGESTIONS ----------------
        st.divider()
        st.subheader("💡 Resume Improvement Suggestions")

        if missing_skills:
            st.warning("Focus on improving these skills:")
            for skill in missing_skills:
                st.write(f"🔹 Add projects or certifications related to **{skill}**")
        else:
            st.success("You have all required technical skills!")

        if text_match_percentage < 50:
            st.error("Your resume content is not well aligned with the job description.")
            st.write("- Use similar keywords from the job description.")
            st.write("- Highlight relevant experience clearly.")
            st.write("- Reduce unrelated content.")
        elif text_match_percentage < 75:
            st.warning("Your resume is moderately aligned.")
            st.write("- Improve project explanations.")
            st.write("- Add measurable achievements.")
        else:
            st.success("Your resume is strongly aligned with this role!")
            st.write("- Maintain clean formatting.")
            st.write("- Add quantified achievements for stronger impact.")