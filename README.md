# Resume Screening & Candidate Ranking System

An ML-powered, ATS-style Resume Screening and Candidate Ranking System that automatically analyzes, scores, and ranks candidates based on a recruiter-defined job description and required skills. The system focuses on explainable AI for hiring, enabling recruiters to clearly understand why a candidate is classified as a strong fit, potential fit, or low fit. This project simulates how modern HR-tech and recruitment platforms perform large-scale resume screening using Natural Language Processing (NLP) and interactive decision-support dashboards.

## Live Deployment
👉 Deployed Application Link: [https://resume-screening-ats-9hjaectpyadtcct5skgku6.streamlit.app/]

## Demo Video
📽 Project Walkthrough / Screen Recording: [https://drive.google.com/file/d/1SyGB2ibfAWFlYnSMQYFMWWbVFOHF8Xbv/view?usp=sharing]

The demo video showcases resume uploads, recruiter input of job descriptions and skills, step-wise analysis with progress indicators, candidate ranking, explainable insights, and downloadable results.

## Key Features
- Resume parsing with support for TXT and PDF formats  
- Semantic resume–job description matching using NLP (TF-IDF and cosine similarity)  
- Explicit skill matching with missing skill gap analysis  
- Composite scoring for fair and transparent candidate ranking  
- ATS-style interactive dashboard built using Streamlit  
- Explainable results, including matched and missing skills  
- Step-wise processing with live progress indicators and status updates  
- Toast notifications for system feedback  
- Downloadable CSV report of ranked candidates  
- Modern dark-themed recruiter-focused user interface  

## How the System Works
Recruiters enter a job description and specify required skills. Resumes are cleaned and preprocessed using NLP techniques. Semantic similarity between each resume and the job description is computed. Skill overlap and missing skills are identified. A composite score combining semantic relevance and skill match ratio is calculated. Candidates are ranked based on this score and results are presented in an explainable, interactive dashboard.

## Scoring Logic
The final candidate ranking is computed using a composite scoring strategy:
Final Score = (Semantic Similarity Weight × Similarity Score) + (Skill Match Weight × Skill Match Ratio)
This ensures balanced decision-making by considering both contextual relevance and explicit skill requirements.

## Tech Stack
Python, Streamlit, scikit-learn, NLTK, pdfplumber, Pandas, NumPy

## Notebook: Model Development & Analysis
The notebook included in the notebooks/ directory documents the complete experimentation and model development process. It covers dataset exploration, text preprocessing decisions, skill extraction logic, similarity analysis, visualizations, and design reasoning behind the final ATS system. The notebook represents the research and analytical foundation, while the Streamlit app represents the production-style implementation.

## Use Cases
HR-tech platforms, resume screening automation tools, recruitment decision-support systems, NLP and ML portfolio projects, and hiring workflow prototypes.

## Disclaimer
This project is a production-ready prototype built for educational and demonstration purposes. It can be extended with embedding-based models, databases, authentication systems, cloud storage, and large language model integrations for enterprise-scale deployment.

## Author
Tejas R M

If you find this project useful, consider starring the repository.
