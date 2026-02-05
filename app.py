import streamlit as st
import pandas as pd
import re
import time
import pdfplumber
import nltk
nltk.download("stopwords")
nltk.download("wordnet")


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Page config
st.set_page_config(
    page_title="Resume Screening System",
    layout="wide"
)


st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: #e5e7eb;
}
.css-1d391kg {
    background-color: #020617;
}
.section-card {
    background: #020617;
    padding: 28px;
    border-radius: 18px;
    margin-bottom: 28px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.45);
}
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 14px;
}
.metric-card {
    background: #020617;
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0 10px 25px rgba(0,0,0,0.45);
}
.metric-card h2 {
    margin: 0;
    color: #38bdf8;
}
.metric-card p {
    margin: 0;
    color: #94a3b8;
}
.stButton>button {
    background: linear-gradient(90deg, #6366f1, #3b82f6);
    color: white;
    border-radius: 10px;
    font-weight: 600;
}
[data-testid="stDataFrame"] {
    background-color: #020617;
    border-radius: 14px;
}
.streamlit-expanderHeader {
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="section-card">
  <div class="section-title">📄 Resume Screening & Candidate Ranking System</div>
  <p style="color:#94a3b8;">
    ML-powered, multi-domain ATS-style dashboard for screening resumes,
    matching recruiter-defined job descriptions, and ranking candidates
    with explainable scoring.
  </p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("data/Resume.csv")

df = load_data()


if "uploaded_resumes" not in st.session_state:
    st.session_state.uploaded_resumes = []

st.sidebar.subheader("📤 Upload Resumes (TXT / PDF)")
uploaded_files = st.sidebar.file_uploader(
    "Upload resumes",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        text = ""
        if file.name.endswith(".txt"):
            text = file.read().decode("utf-8", errors="ignore")
        elif file.name.endswith(".pdf"):
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""

        if len(text.strip()) > 300:
            st.session_state.uploaded_resumes.append({
                "ID": f"UPLOAD_{len(st.session_state.uploaded_resumes)}",
                "Resume_str": text,
                "Category": "Uploaded"
            })

if st.sidebar.button("🗑 Clear Uploaded Resumes"):
    st.session_state.uploaded_resumes = []

if st.session_state.uploaded_resumes:
    df = pd.concat(
        [df, pd.DataFrame(st.session_state.uploaded_resumes)],
        ignore_index=True
    )


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

df["clean_resume"] = df["Resume_str"].apply(clean_text)


st.sidebar.header("🧾 Recruiter Inputs")

job_description = st.sidebar.text_area("Job Description", height=200)
skills_input = st.sidebar.text_input(
    "Required Skills (comma-separated)",
    "python, machine learning, sql"
)
search_keyword = st.sidebar.text_input("🔎 Search keyword (optional)")
top_n = st.sidebar.slider("Candidates to display", 5, 50, 10, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Scoring Weights")
similarity_weight = st.sidebar.slider("Semantic Similarity", 0.0, 1.0, 0.7)
skill_weight = 1 - similarity_weight

analyze_btn = st.sidebar.button("🔍 Analyze Resumes")


def short_text(text, n=60):
    return text[:n] + "..." if len(text) > n else text

def decision_label(score):
    if score >= 0.25:
        return "🟢 Strong Fit"
    elif score >= 0.15:
        return "🟡 Potential"
    return "🔴 Low Fit"

def highlight(text, skills):
    for s in skills:
        text = re.sub(fr"\b({s})\b", r"**\1**", text, flags=re.I)
    return text


if analyze_btn and job_description.strip():

    st.toast("🚀 Resume analysis started", icon="⏳")

    progress = st.progress(0)
    status = st.empty()

    with st.spinner("Analyzing resumes…"):

        status.markdown("🔹 **Parsing resumes…**")
        time.sleep(0.4)
        progress.progress(20)

        skills = [s.strip().lower() for s in skills_input.split(",")]

        status.markdown("🔹 **Matching required skills…**")
        df["matched_skills"] = df["clean_resume"].apply(lambda x: [s for s in skills if s in x])
        df["missing_skills"] = df["matched_skills"].apply(lambda x: list(set(skills) - set(x)))
        df["skill_ratio"] = df["matched_skills"].apply(len) / len(skills)
        time.sleep(0.4)
        progress.progress(45)

        status.markdown("🔹 **Computing semantic similarity…**")
        tfidf = TfidfVectorizer().fit_transform(
            df["clean_resume"].tolist() + [clean_text(job_description)]
        )
        df["similarity_score"] = cosine_similarity(tfidf[:-1], tfidf[-1:]).flatten()
        time.sleep(0.4)
        progress.progress(70)

        status.markdown("🔹 **Ranking candidates…**")
        df["final_score"] = similarity_weight * df["similarity_score"] + skill_weight * df["skill_ratio"]
        df["Decision"] = df["final_score"].apply(decision_label)
        df_ranked = df.sort_values("final_score", ascending=False).reset_index(drop=True)
        df_ranked["Rank"] = df_ranked.index + 1
        df_ranked["preview"] = df_ranked["Resume_str"].apply(short_text)
        time.sleep(0.4)
        progress.progress(90)

        status.markdown("🔹 **Finalizing results…**")
        time.sleep(0.3)
        progress.progress(100)

    status.empty()
    st.toast("✅ Analysis complete!", icon="🎉")

    if search_keyword:
        df_ranked = df_ranked[
            df_ranked["Resume_str"].str.lower().str.contains(search_keyword.lower())
        ]

    display_df = df_ranked.head(top_n)

    
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Hiring Snapshot</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='metric-card'><h2>{len(df_ranked)}</h2><p>Total</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><h2>{(df_ranked.Decision=='🟢 Strong Fit').sum()}</h2><p>Strong Fit</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><h2>{(df_ranked.Decision=='🟡 Potential').sum()}</h2><p>Potential</p></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='metric-card'><h2>{(df_ranked.Decision=='🔴 Low Fit').sum()}</h2><p>Low Fit</p></div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏆 Candidate Evaluation</div>', unsafe_allow_html=True)

    left, right = st.columns([1.5, 1])
    with left:
        st.dataframe(display_df[["Rank", "preview", "Decision", "final_score"]], use_container_width=True)
    with right:
        st.markdown("#### 📈 Distribution")
        st.scatter_chart(df_ranked, x="similarity_score", y="skill_ratio")

    st.markdown('</div>', unsafe_allow_html=True)

   
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 Candidate Details</div>', unsafe_allow_html=True)

    for _, r in display_df.iterrows():
        with st.expander(f"Rank {r.Rank} – {r.preview} ({r.Decision})"):
            st.progress(min(r.final_score, 1.0))
            st.write("**Matched Skills:**", r.matched_skills)
            st.write("**Missing Skills:**", r.missing_skills)
            st.markdown(highlight(r.Resume_str, skills))

    st.markdown('</div>', unsafe_allow_html=True)

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Results", csv, "ranked_candidates.csv", "text/csv")

else:
    st.markdown("""
    <div class="section-card">
      <b>👉 How to use:</b><br>
      1. Enter Job Description<br>
      2. Add Required Skills<br>
      3. Adjust scoring weights<br>
      4. Click <b>Analyze Resumes</b>
    </div>
    """, unsafe_allow_html=True)
