import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import PyPDF2
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import os
from pathlib import Path
import base64

# --- Path Setup ---
BASE_DIR = Path(__file__).parent
CSS_PATH = BASE_DIR / "assets" / "style.css"
IMAGE_PATH = BASE_DIR / "images" / "header_banner.png"

# --- Page Config ---
st.set_page_config(
    page_title="AI Resume Matcher Pro",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Injection ---
def inject_css():
    try:
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"CSS Error: {str(e)}")
        # Fallback inline CSS
        st.markdown("""
        <style>
            .stApp { background: #f0f2f6 !important; }
            h1 { color: #4D44DB !important; }
        </style>
        """, unsafe_allow_html=True)

# --- Header Image ---
def load_header():
    # Load image as base64
    with open(str(IMAGE_PATH), "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    image_data = f"data:image/png;base64,{encoded}"
    st.markdown(
        f"""
        <div style="
            display: flex; 
            flex-direction: row; 
            align-items: center; 
            justify-content: center; 
            gap: 40px; 
            padding: 36px 0 8px 0;">
            <img src="{image_data}" alt="AI Resume Matcher" 
                style="width: 420px; height: 220px; border-radius: 24px; box-shadow: 0 8px 32px 0 rgba(76, 76, 255, 0.12); object-fit: cover; background: #fff4fa;">
            <div style="font-size: 2.1rem; color: #2B2557; font-weight: 700; font-family: 'Segoe UI', 'Helvetica Neue', sans-serif; line-height: 1.2; max-width: 520px;">
                Unlock the power of AI for            smart,<br>seamless CV &amp; job matching
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )


inject_css()
load_header()



# --- Rest of your existing code ---
embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 🔐 Configure Gemini
with open("gemini_api_key.txt") as f:
    api_key = f.read().strip()
genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-1.5-flash")

def requirement_based_matching(resume_text, job_text):
    """
    New matching function that evaluates specific requirements against the resume
    Returns a dictionary with scores for each requirement
    """
    # This is where we'll implement the logic shown in your screenshots
    # For now, I'll create a placeholder that mimics your example
    
    prompt = f"""
Analyze this resume against the job description and provide scores for each key requirement.
Use the exact scoring format shown in the example below.

Example format:
Strong attention to detail and ability to identify hazards: 8/10
Excellent communication and interpersonal skills: 9/10
Teamwork and independent work capability: 10/10
Proficiency in Python or AI programming languages: 10/10
Experience with machine learning: 5/10

Resume:
{resume_text}

Job Description:
{job_text}

Provide the analysis in the exact same format as the example, with one requirement per line followed by colon and score.
"""
    
    response = model.generate_content(prompt)
    return parse_requirement_scores(response.text)

def parse_requirement_scores(analysis_text):
    """
    Parse the AI-generated analysis into a structured format
    """
    lines = [line.strip() for line in analysis_text.split('\n') if line.strip()]
    requirements = {}
    
    for line in lines:
        if ':' in line:
            req, score = line.split(':', 1)
            req = req.strip()
            score = score.strip()
            
            # Extract numeric score (handle cases like "8/10")
            if '/' in score:
                score = score.split('/')[0].strip()
            try:
                score = int(score)
            except ValueError:
                score = 0
                
            requirements[req] = score
    
    return requirements


def extract_personal_info(text):
    """
    Extract name, email, and phone number from CV text
    """
    # Extract name (simple approach - first line often contains name)
    name = "the candidate"
    name_match = re.search(r"^(.*?)\n", text)
    if name_match:
        name = name_match.group(1).strip()
    
    # Extract email
    email = ""
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    if email_match:
        email = email_match.group(0)
    
    # Extract phone number (international format)
    phone = ""
    phone_match = re.search(r'(\+?\d[\d\s\-\(\)]{7,}\d)', text)
    if phone_match:
        phone = phone_match.group(1).strip()
    
    return {
        'name': name,
        'email': email,
        'phone': phone
    }


    

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
# Function to extract skills section (simple regex or fallback)
def extract_skills(text):
    match = re.search(r"(skills|competencies|technologies).*?:\s*(.*)", text, re.IGNORECASE)
    return match.group(2) if match else text

# Function to extract experience section (simple regex or fallback)
def extract_experience(text):
    match = re.search(r"(experience|work history|internships).*?:\s*(.*)", text, re.IGNORECASE)
    return match.group(2) if match else text

# Function to calculate weighted similarity based on selected weights
def weighted_similarity(resume_text, job_text, skills_weight, experience_weight):
    # Extract skills and experience parts
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)

    resume_experience = extract_experience(resume_text)
    job_experience = extract_experience(job_text)

    # Embed each part
    resume_skills_emb = embedder.encode([resume_skills])
    job_skills_emb = embedder.encode([job_skills])

    resume_experience_emb = embedder.encode([resume_experience])
    job_experience_emb = embedder.encode([job_experience])

    # Calculate similarities
    skills_similarity = cosine_similarity(resume_skills_emb, job_skills_emb)[0][0]
    experience_similarity = cosine_similarity(resume_experience_emb, job_experience_emb)[0][0]

    # Weighted score
    final_score = (skills_weight * skills_similarity) + (experience_weight * experience_similarity)

    return final_score
    

# HR Interface
if interface_type == "HR Interface":
    st.title("💼 AI Resume Matcher for HR 🔮")
    st.markdown("Upload a job description and candidate's CV to get a detailed requirement analysis.")

    # Choose input type for the job description
    job_input_type = st.selectbox(
        "How do you want to enter the job description?",
        ("Upload Text File", "Upload PDF", "Write Manually")
    )

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
    cv_input_type = st.selectbox(
        "How do you want to enter the candidate's CV?",
        ("Upload Text File", "Upload PDF", "Write Manually")
    )

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
        # Extract candidate info
        candidate_info = extract_personal_info(resume_text)
        
        with st.spinner("Analyzing requirements match..."):
            requirements_scores = requirement_based_matching(resume_text, job_text)
            
        # Calculate overall score
        if requirements_scores:
            overall_score = sum(requirements_scores.values()) / len(requirements_scores)
            overall_percentage = (overall_score / 10) * 100
            
            st.subheader(f"Overall Match Score: {overall_percentage:.0f}%")
            
            # Display candidate information
            st.subheader("Candidate Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Name:** {candidate_info['name']}")
            with col2:
                st.markdown(f"**Email:** {candidate_info['email'] if candidate_info['email'] else 'Not found'}")
            with col3:
                st.markdown(f"**Phone:** {candidate_info['phone'] if candidate_info['phone'] else 'Not found'}")
            
            # Display requirements breakdown (minimal, clean)
            st.subheader("Requirement Breakdown")
            for requirement, score in requirements_scores.items():
                st.markdown(
                    f"<div style='font-weight:700; color:#18104c; margin-bottom:0.15em; font-size:1.12rem;'>{requirement}</div>",
                    unsafe_allow_html=True
                )
                st.progress(score / 10)
                st.markdown(
                    f"<div style='font-size:0.97rem; color:#666; margin-bottom:1.3em;'>Score: {score}/10</div>",
                    unsafe_allow_html=True
                )
            # Add personalized summary feedback
            st.subheader("Overall Feedback")
            if overall_score >= 8:
                st.success(f"{candidate_info['name']} is a very strong fit for this role!")
            elif overall_score >= 6:
                st.warning(f"{candidate_info['name']} is a good fit but could improve in some areas.")
            else:
                st.error(f"There are significant gaps between {candidate_info['name']}'s profile and the job requirements.")
            
            # Add contact information reminder
            if candidate_info['email'] or candidate_info['phone']:
                st.info(
                    f"Contact information: "
                    f"{'Email: ' + candidate_info['email'] if candidate_info['email'] else ''} "
                    f"{'Phone: ' + candidate_info['phone'] if candidate_info['phone'] else ''}"
                )

elif interface_type == "Candidate Interface":
    st.title("👩‍💼 AI Resume Improvement Assistant")
    st.markdown("Get detailed feedback on how your CV matches specific job requirements.")

    # Choose input type for the CV
    resume_input_type = st.selectbox("How do you want to enter your CV?", 
                                   ("Upload Text File", "Upload PDF", "Write Manually"))

    # CV handling based on chosen input type
    resume_text = ""  # Changed from candidate_cv to resume_text for consistency
    if resume_input_type == "Upload Text File":
        cv_file = st.file_uploader("📄 Upload your CV (.txt)", type=['txt'])
        if cv_file:
            resume_text = cv_file.read().decode('utf-8')
    elif resume_input_type == "Upload PDF":
        cv_pdf = st.file_uploader("📄 Upload your CV (.pdf)", type=['pdf'])
        if cv_pdf:
            resume_text = extract_text_from_pdf(cv_pdf)
    elif resume_input_type == "Write Manually":
        resume_text = st.text_area("✍️ Write your CV manually:")

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
    if resume_text and job_text:  # Changed from candidate_cv to resume_text
        # Extract candidate info
        candidate_info = extract_personal_info(resume_text)
        
        with st.spinner("Analyzing your profile against job requirements..."):
            requirements_scores = requirement_based_matching(resume_text, job_text)
            
        if requirements_scores:
            overall_score = sum(requirements_scores.values()) / len(requirements_scores)
            
            st.subheader("Your Match Breakdown")
            
            for requirement, score in requirements_scores.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{requirement}**")
                with col2:
                    st.markdown(f"**{score}/10**")
                
                # Color code based on score
                if score >= 8:
                    st.success("✓ Strong match")
                elif score >= 5:
                    st.warning("△ Partial match")
                else:
                    st.error("✕ Weak match")
                
                st.write("")
            
            # Generate improvement advice
            with st.spinner("Generating personalized improvement advice..."):
                missing_skills = [req for req, score in requirements_scores.items() if score < 6]
                if missing_skills:
                    advice = generate_candidate_advice("\n".join(missing_skills))
                    st.subheader("Improvement Advice")
                    st.info(advice)
                
                # Generate motivation letter for good matches
                if overall_score >= 6:
                    st.subheader("Suggested Motivation Letter")
                    motivation_letter = generate_motivation_letter(resume_text, job_text)
                    st.success(motivation_letter)


st.markdown(
    "<div style='text-align:right; font-size:0.95rem; color:#b8b8b8; margin-top:18px;'>"
    "Built by Eya Bchir · Powered by Streamlit & Gemini API"
    "</div>", unsafe_allow_html=True
)
                    

