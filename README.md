# AI Resume Matcher

## Overview
This project is an AI-based tool that matches job descriptions with resumes using natural language processing (NLP) models. It provides HR insights into the compatibility of a candidate with a job description and generates personalized advice and motivation letters for candidates based on the matching.

## Features
- Resume and Job Description matching using cosine similarity.
- Missing skills and qualifications identification for candidates.
- Motivation letter generation for candidates.
- Advice generation for candidates based on missing qualifications.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/<your-username>/AI-Resume-Matcher.git
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. To run the project, you can use the following command:
    ```bash
    streamlit run app.py
    ```

## Requirements
- Python 3.x
- `streamlit`
- `sentence-transformers`
- `sklearn`
- `google-generativeai`
- `PyPDF2`

## How to Use
- **HR Interface**: Upload a job description and a candidate's CV to analyze the match and receive insights about the qualifications and skills.
- **Candidate Interface**: Upload your CV and a job description to get a match score, motivation letter, and advice on missing qualifications.

## Acknowledgments
- Hugging Face for the Sentence Transformers model.
- Google for the Gemini API.
