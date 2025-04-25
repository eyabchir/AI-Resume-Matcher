# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python (env1)
#     language: python
#     name: env1
# ---

# %%

# %%

# %%

# %%

# %%

# %%

# %%
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import PyPDF2
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# 🎯 Load Sentence Transformer model
embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 🔐 Configure Gemini
with open("gemini_api_key.txt") as f:
    api_key = f.read().strip()
genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

# 🪄 Explanation function for HR Interface
def get_match_explanation(resume, job):
    prompt = f"""
You are an AI career advisor. Given a resume and a job description, explain why this job is a good match for the person. Be specific and mention relevant skills, experience, or tools that align.

Resume:
{resume}

Job Description:
{job}

Response:
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# 🪄 Missing skills function for HR Interface
def get_missing_skills(resume, job):
    prompt = f"""
You are an AI career advisor. Given a resume and a job description, identify and list the skills, qualifications, or experience that the resume is missing based on the job description.

Resume:
{resume}

Job Description:
{job}

Response:
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# 🪄 Motivation Letter Generator
def generate_motivation_letter(resume, job):
    prompt = f"""
You are an AI career advisor. Given a resume and a job description, generate a personalized motivation letter for the candidate. Highlight their skills and experience that align well with the job.

Resume:
{resume}

Job Description:
{job}

Response:
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# 🪄 Candidate Advice for Missing Qualifications
def generate_candidate_advice(missing_skills):
    prompt = f"""
You are an AI career advisor. Based on the following missing skills, provide advice to the candidate on how they can work on these qualifications to improve their chances for the job.

Missing Skills:
{missing_skills}

Response:
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# 🪄 Function to extract text from a PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# 🌟 Streamlit UI for selecting the interface type
interface_type = st.sidebar.selectbox("Select Interface", ["HR Interface", "Candidate Interface"])

# Function to preprocess text
def preprocess_text(text):
    """Preprocess text by removing special characters, extra spaces, and stopwords."""
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)    # Remove multiple spaces
    text = text.lower()                 # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])  # Remove stopwords
    return text

# HR Interface
if interface_type == "HR Interface":
    st.title("💼 AI Resume Matcher for HR 🔮")
    st.markdown("Upload a job description and candidate's CV to match and get insights.")

    # Choose input type for the job description
    job_input_type = st.selectbox("How do you want to enter the job description?", 
                                  ("Upload Text File", "Upload PDF", "Write Manually"))

    # Job description handling based on chosen input type
    job_text = ""
    if job_input_type == "Upload Text File":
        job_file = st.file_uploader("📂 Upload job description (.txt)", type=['txt'])
        if job_file:
            job_text = job_file.read().decode('utf-8')

    elif job_input_type == "Upload PDF":
        job_pdf = st.file_uploader("📂 Upload job description (.pdf)", type=['pdf'])
        if job_pdf:
            job_text = extract_text_from_pdf(job_pdf)

    elif job_input_type == "Write Manually":
        job_text = st.text_area("✍️ Write job description manually:")

    # Choose input type for the CV
    cv_input_type = st.selectbox("How do you want to enter the candidate's CV?", 
                                 ("Upload Text File", "Upload PDF", "Write Manually"))

    # CV handling based on chosen input type
    resume_text = ""
    if cv_input_type == "Upload Text File":
        resume_file = st.file_uploader("📄 Upload candidate's CV (.txt)", type=['txt'])
        if resume_file:
            resume_text = resume_file.read().decode('utf-8')

    elif cv_input_type == "Upload PDF":
        resume_pdf = st.file_uploader("📄 Upload candidate's CV (.pdf)", type=['pdf'])
        if resume_pdf:
            resume_text = extract_text_from_pdf(resume_pdf)

    elif cv_input_type == "Write Manually":
        resume_text = st.text_area("✍️ Write candidate's CV manually:")

    # Process matching when both job description and resume are provided
    if resume_text and job_text:
        # Preprocess both the resume and job description
        resume_text = preprocess_text(resume_text)
        job_text = preprocess_text(job_text)

        # Compute similarity
        with st.spinner("Analyzing match..."):
            resume_emb = embedder.encode([resume_text])
            job_emb = embedder.encode([job_text])
            similarity_score = cosine_similarity(resume_emb, job_emb)[0][0]

        # Show match result
        st.subheader(f"Matching Score: {similarity_score * 100:.2f}%")
        
        if similarity_score > 0.5:
            # Score > 50%: Provide match explanation
            with st.spinner("Generating match explanation..."):
                explanation = get_match_explanation(resume_text, job_text)
            st.markdown("**💡 Match Explanation:**")
            st.success(explanation)
        else:
            # Score < 50%: List missing skills
            with st.spinner("Identifying missing skills..."):
                missing_skills = get_missing_skills(resume_text, job_text)
            st.markdown("**❗ Missing Skills and Qualifications:**")
            st.error(missing_skills)

# Candidate Interface
elif interface_type == "Candidate Interface":
    st.title("👩‍💼 AI Resume Improvement Assistant")
    st.markdown("Here we will help you understand how to improve your CV for a better match with job descriptions.")

    # Choose input type for the CV
    resume_input_type = st.selectbox("How do you want to enter your CV?", 
                                     ("Upload Text File", "Upload PDF", "Write Manually"))

    # CV handling based on chosen input type
    candidate_cv = ""
    if resume_input_type == "Upload Text File":
        cv_file = st.file_uploader("📄 Upload your CV (.txt)", type=['txt'])
        if cv_file:
            candidate_cv = cv_file.read().decode('utf-8')

    elif resume_input_type == "Upload PDF":
        cv_pdf = st.file_uploader("📄 Upload your CV (.pdf)", type=['pdf'])
        if cv_pdf:
            candidate_cv = extract_text_from_pdf(cv_pdf)

    elif resume_input_type == "Write Manually":
        candidate_cv = st.text_area("✍️ Write your CV manually:")

    # Choose input type for the job description
    job_input_type = st.selectbox("How do you want to enter the job description?", 
                                  ("Upload Text File", "Upload PDF", "Write Manually"))

    # Job description handling based on chosen input type
    job_text = ""
    if job_input_type == "Upload Text File":
        job_file = st.file_uploader("📂 Upload job description (.txt)", type=['txt'])
        if job_file:
            job_text = job_file.read().decode('utf-8')

    elif job_input_type == "Upload PDF":
        job_pdf = st.file_uploader("📂 Upload job description (.pdf)", type=['pdf'])
        if job_pdf:
            job_text = extract_text_from_pdf(job_pdf)

    elif job_input_type == "Write Manually":
        job_text = st.text_area("✍️ Write job description manually:")

    # Process matching when both CV and job description are provided
    if candidate_cv and job_text:
        # Preprocess both the resume and job description
        candidate_cv = preprocess_text(candidate_cv)
        job_text = preprocess_text(job_text)

        # Compute similarity
        with st.spinner("Analyzing match..."):
            resume_emb = embedder.encode([candidate_cv])
            job_emb = embedder.encode([job_text])
            similarity_score = cosine_similarity(resume_emb, job_emb)[0][0]

        # Show match result
        st.subheader(f"Matching Score: {similarity_score * 100:.2f}%")
        
        if similarity_score > 0.5:
            # Score > 50%: Generate motivation letter
            with st.spinner("Generating motivation letter..."):
                motivation_letter = generate_motivation_letter(candidate_cv, job_text)
            st.markdown("**💌 Motivation Letter:**")
            st.success(motivation_letter)
        else:
            # Score < 50%: Provide advice on missing qualifications
            with st.spinner("Identifying missing skills..."):
                missing_skills = get_missing_skills(candidate_cv, job_text)
            st.markdown("**❗ Missing Skills and Qualifications:**")
            st.error(missing_skills)

            with st.spinner("Generating advice for the candidate..."):
                candidate_advice = generate_candidate_advice(missing_skills)
            st.markdown("**💡 Candidate's Improvement Advice:**")
            st.success(candidate_advice)


# %%

# %%

# %%

# %%

# %%

# %%

# %%
