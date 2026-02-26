import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AI Interview Prep System", layout="centered")

st.title("AI Interview Preparation System")
st.write("Upload your resume and get evaluated with AI-powered feedback.")

# -----------------------------
# Skill Keywords
# -----------------------------
skill_list = [
    "python", "machine learning", "ml", "deep learning",
    "data science", "aws", "azure", "cloud",
    "networking", "computer networks"
]

question_bank = {
    "python": [
        "Explain OOP in Python.",
        "What is the difference between list and tuple?"
    ],
    "machine learning": [
        "What is overfitting?",
        "Explain bias-variance tradeoff."
    ],
    "data science": [
        "What is data preprocessing?",
        "Explain normalization."
    ],
    "cloud": [
        "What is scalability in cloud computing?",
        "Explain IaaS vs PaaS."
    ],
    "networking": [
        "What is jitter?",
        "Explain packet loss."
    ]
}

ideal_answers = {
    "Explain OOP in Python.": "Object oriented programming is based on classes and objects with concepts like inheritance and polymorphism.",
    "What is the difference between list and tuple?": "List is mutable while tuple is immutable.",
    "What is overfitting?": "Overfitting occurs when a model performs well on training data but poorly on new data.",
    "Explain bias-variance tradeoff.": "Bias is error due to wrong assumptions and variance is error due to sensitivity to small changes.",
    "What is data preprocessing?": "It involves cleaning and transforming raw data into usable format.",
    "Explain normalization.": "Normalization scales data to a standard range.",
    "What is scalability in cloud computing?": "Scalability is the ability to increase or decrease resources based on demand.",
    "Explain IaaS vs PaaS.": "IaaS provides infrastructure while PaaS provides development platform.",
    "What is jitter?": "Jitter is variation in packet delay.",
    "Explain packet loss.": "Packet loss occurs when packets fail to reach destination."
}

# -----------------------------
# Resume Text Extraction
# -----------------------------
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text.lower()

# -----------------------------
# Skill Detection
# -----------------------------
def detect_skills(text):
    detected = []
    for skill in skill_list:
        if skill in text:
            detected.append(skill)
    return list(set(detected))

# -----------------------------
# Map Skills to Categories
# -----------------------------
def map_skill_to_category(skill):
    if skill in ["ml", "deep learning"]:
        return "machine learning"
    elif skill in ["aws", "azure"]:
        return "cloud"
    elif skill in ["computer networks"]:
        return "networking"
    else:
        return skill

# -----------------------------
# Similarity Calculation
# -----------------------------
def calculate_similarity(user_answer, ideal_answer):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([user_answer, ideal_answer])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100, 2)

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])

if uploaded_file:

    resume_text = extract_text_from_pdf(uploaded_file)
    skills = detect_skills(resume_text)

    st.subheader("Detected Skills")

    if skills:
        st.success(", ".join(skills))
    else:
        st.warning("No matching predefined skills found. Using default Python questions.")
        skills = ["python"]

    # -----------------------------
    # Generate Questions
    # -----------------------------
    questions = []
    for skill in skills:
        category = map_skill_to_category(skill)
        questions.extend(question_bank.get(category, []))

    questions = list(set(questions))  # remove duplicates

    st.subheader("Interview Questions")

    scores = []

    for question in questions:
        st.markdown(f"**{question}**")
        user_answer = st.text_area("Your Answer:", key=question)

        if user_answer:
            similarity_score = calculate_similarity(user_answer, ideal_answers[question])
            sentiment_score = TextBlob(user_answer).sentiment.polarity

            st.write(f"Similarity Score: **{similarity_score}%**")
            st.write(f"Confidence (Sentiment Polarity): **{round(sentiment_score,2)}**")

            # Feedback
            if similarity_score > 75:
                st.success("Excellent answer! Strong conceptual clarity.")
            elif similarity_score > 50:
                st.info("Good answer, but you can improve explanation depth.")
            else:
                st.warning("Answer needs improvement. Try adding more technical details.")

            scores.append(similarity_score)

    # -----------------------------
    # Final Score
    # -----------------------------
    if scores:
        final_score = sum(scores) / len(scores)

        st.subheader("Final Performance Score")
        st.success(f"{round(final_score,2)} %")

        # Performance Category
        if final_score > 75:
            level = "Strong Candidate"
        elif final_score > 50:
            level = "Average Candidate"
        else:
            level = "Needs Improvement"

        st.write(f"Performance Level: **{level}**")

        # Download Report
        st.download_button(
            "Download Report",
            data=f"Final Score: {round(final_score,2)}%\nPerformance Level: {level}",
            file_name="Interview_Report.txt"
        )

        # Graph
        fig, ax = plt.subplots()
        ax.bar(["Performance"], [final_score])
        ax.set_ylim(0, 100)
        st.pyplot(fig)