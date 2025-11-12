import os
import re
import sqlite3
from collections import Counter
import streamlit as st
from dotenv import load_dotenv
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import nltk
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import spacy
from fpdf import FPDF

# Optional OCR imports
try:
    from pdf2image import convert_from_bytes
    from PIL import Image
    import pytesseract
    PDF2IMAGE_AVAILABLE = True
except Exception:
    PDF2IMAGE_AVAILABLE = False

# -------------------- CONFIG --------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
DB_PATH = os.path.join(os.getcwd(), "sra.db")
nlp = spacy.load("en_core_web_sm")

# -------------------- STYLES --------------------
st.markdown("""
<style>
body {background-color: #0e1117;}
.main-title {
    text-align:center; color:#00E0C6; font-size:48px; font-weight:800; margin-bottom:0;
    text-shadow: 0px 0px 12px #00ADB5;
}
.sub-title {
    text-align:center; color:#CCCCCC; font-size:18px; margin-top:4px;
}
.card {
    background-color:rgba(30,30,30,0.7);
    backdrop-filter:blur(8px);
    padding:20px;
    border-radius:12px;
    color:#EEE;
    box-shadow: 0 4px 20px rgba(0,173,181,0.3);
}
.metric-box {
    background: linear-gradient(145deg, #1f1f1f, #2b2b2b);
    padding:18px;
    border-radius:15px;
    text-align:center;
    color:#EEE;
    box-shadow: inset 0 0 8px #00ADB5, 0 0 10px rgba(0,0,0,0.6);
}
.badge {
    display:inline-block;
    padding:6px 10px;
    margin:2px;
    border-radius:10px;
    font-size:13px;
    color:#fff;
    font-weight:500;
}
.badge-matched {background-color:#00ADB5;}
.badge-missing {background-color:#FF5722;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>AI Resume Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Insightful Visual Analysis for Job Fit, Skills, and Performance</p>", unsafe_allow_html=True)

# -------------------- UTILITIES --------------------
def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True) if os.path.dirname(DB_PATH) else None
    return sqlite3.connect(DB_PATH)

def create_tables():
    with get_connection() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            resume_text TEXT,
            job_desc TEXT
        )""")
        conn.commit()

def save_resume(name, resume_text, job_desc):
    with get_connection() as conn:
        conn.execute("INSERT INTO resumes (name, resume_text, job_desc) VALUES (?, ?, ?)", (name, resume_text, job_desc))
        conn.commit()

# Try pdfminer first, then fallback to OCR using pdf2image + pytesseract if available.
def extract_pdf_text(uploaded_file):
    if uploaded_file is None:
        return ""
    try:
        uploaded_file.seek(0)
        raw_text = extract_text(uploaded_file)
        if raw_text and raw_text.strip():
            return raw_text.strip()
    except Exception:
        # ignore and try OCR below
        pass

    # OCR fallback for scanned PDFs
    if PDF2IMAGE_AVAILABLE:
        try:
            uploaded_file.seek(0)
            pdf_bytes = uploaded_file.read()
            images = convert_from_bytes(pdf_bytes, dpi=200)
            text_pages = []
            for img in images:
                # convert to RGB to ensure compatibility
                if img.mode != "RGB":
                    img = img.convert("RGB")
                page_text = pytesseract.image_to_string(img)
                text_pages.append(page_text)
            ocr_text = "\n".join(text_pages).strip()
            return ocr_text
        except Exception:
            return ""
    else:
        # If pdf2image or pytesseract not available, return empty string
        return ""

@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def calculate_similarity(text1, text2):
    model = load_model()
    emb1 = model.encode([text1 or ""], normalize_embeddings=True)
    emb2 = model.encode([text2 or ""], normalize_embeddings=True)
    sim = cosine_similarity(emb1, emb2)[0][0]
    return max(0.0, min(1.0, float(sim)))

def get_groq_client():
    if not API_KEY:
        st.warning("GROQ API key not found in environment.")
        return None
    return Groq(api_key=API_KEY)

def generate_report(resume, job_desc):
    client = get_groq_client()
    if not client:
        return "GROQ API key missing. AI report not generated."
    prompt = f"""
You are an AI Resume Analyzer.
Compare the resume and job description below.
1. Evaluate job match and assign a score (out of 5) with reasoning.
2. Suggest improvements and missing keywords.
3. Highlight tone and professionalism.

Resume: {resume}
---
Job Description: {job_desc}
"""
    try:
        res = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        return (res.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Error generating report: {e}"

def extract_skills(text):
    skills = [
        'python','java','c++','sql','excel','powerbi','ml','ai','communication',
        'leadership','teamwork','project management','javascript','react','django',
        'flask','tensorflow','pandas','data analysis','html','css','aws','git'
    ]
    return sorted({s for s in skills if re.search(rf'\b{re.escape(s)}\b', text or "", re.IGNORECASE)})

def extract_keywords(text):
    try:
        tokens = nltk.word_tokenize(text or "")
        tagged = nltk.pos_tag(tokens)
        nouns = [w.lower() for w, pos in tagged if pos.startswith("NN") and len(w) > 3]
        return [w for w, _ in Counter(nouns).most_common(15)]
    except Exception:
        return []

def extract_resume_entities(text):
    doc = nlp(text)
    entities = {"ORG": [], "DATE": [], "GPE": [], "EDUCATION": []}
    education_keywords = ["B.Tech", "B.E", "M.Tech", "MBA", "BSc", "MSc", "PhD", "Bachelor", "Master"]
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    for word in education_keywords:
        if re.search(word, text, re.IGNORECASE):
            entities["EDUCATION"].append(word)
    return entities

def extract_experience_periods(text):
    # Finds year ranges like 2018-2020 or 2018 – 2020 or "2018 to 2020"
    ranges = re.findall(r'(\b\d{4}\b)[\s\-–to]{1,6}(\b\d{4}\b)', text, flags=re.IGNORECASE)
    data = [{"Start": s, "End": e} for s, e in ranges if int(e) >= int(s)]
    return pd.DataFrame(data)

def generate_pdf_report(report_text, ats, ai_score, skills, missing):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "AI Resume Analyzer Report", ln=True, align="C")
    pdf.set_font("Arial", "", 11)
    pdf.ln(4)
    pdf.multi_cell(0, 7, f"ATS Score: {ats*100:.2f}%")
    pdf.multi_cell(0, 7, f"AI Evaluation Score: {ai_score:.2f}/5")
    pdf.multi_cell(0, 7, f"Matched Skills: {', '.join(skills) or 'None'}")
    pdf.multi_cell(0, 7, f"Missing Keywords: {', '.join(missing) or 'None'}")
    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 7, "AI Generated Report:", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, report_text or "No report generated.")
    file_path = "AI_Resume_Report.pdf"
    pdf.output(file_path)
    return file_path

# -------------------- APP LOGIC --------------------
create_tables()

if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False
if "resume" not in st.session_state:
    st.session_state.resume = ""
if "job_desc" not in st.session_state:
    st.session_state.job_desc = ""

# Input form (UI kept unchanged)
if not st.session_state.form_submitted:
    with st.form("resume_form"):
        st.markdown("### Upload Resume and Job Description")
        c1, c2 = st.columns(2)
        with c1:
            resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
        with c2:
            st.session_state.job_desc = st.text_area("Paste Job Description:", height=200, value=st.session_state.job_desc)
        submit = st.form_submit_button("Analyze Resume")
        if submit:
            if st.session_state.job_desc and resume_file:
                extracted = extract_pdf_text(resume_file)
                if extracted and extracted.strip():
                    st.session_state.resume = extracted
                    st.session_state.form_submitted = True
                    save_resume("Candidate", st.session_state.resume, st.session_state.job_desc)
                    st.rerun()
                else:
                    st.warning("No readable text found in the uploaded PDF. Try using a clear-scanned PDF or ensure OCR dependencies are installed.")
            else:
                st.warning("Please upload both the resume and job description.")

# Analysis and visual dashboard
if st.session_state.form_submitted:
    st.markdown("---")
    with st.spinner("Analyzing resume..."):
        ats_score = calculate_similarity(st.session_state.resume, st.session_state.job_desc)
        report_text = generate_report(st.session_state.resume, st.session_state.job_desc)
        skills = extract_skills(st.session_state.resume)
        jd_keywords = extract_keywords(st.session_state.job_desc)
        missing_keywords = [kw for kw in jd_keywords if kw not in [s.lower() for s in skills]]

    # Top metrics
    m1, m2, m3 = st.columns(3)
    m1.markdown(f"<div class='metric-box'>ATS Similarity<br>{round(ats_score*100,2)}%</div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='metric-box'>AI Evaluation Score<br>{round(ats_score*5,2)}/5</div>", unsafe_allow_html=True)
    total_skills = len(skills) + len(missing_keywords) if (len(skills) + len(missing_keywords)) > 0 else 1
    m3.markdown(f"<div class='metric-box'>Skill Match<br>{len(skills)}/{total_skills}</div>", unsafe_allow_html=True)

    # Expandable AI report
    with st.expander("Detailed AI Report", expanded=True):
        st.markdown(f"<div class='card'>{report_text}</div>", unsafe_allow_html=True)

    # Entities
    with st.expander("Extracted Entities"):
        ents = extract_resume_entities(st.session_state.resume)
        st.write("Organizations:", ", ".join(sorted(set(ents["ORG"]))) or "N/A")
        st.write("Education:", ", ".join(sorted(set(ents["EDUCATION"]))) or "N/A")
        st.write("Dates:", ", ".join(sorted(set(ents["DATE"]))) or "N/A")
        st.write("Locations:", ", ".join(sorted(set(ents["GPE"]))) or "N/A")

    # Visuals arranged compactly with a short text report beside each
    st.subheader("Visual Analysis Dashboard")
    left, right = st.columns([2, 1])

    # Left: main graphs in two columns inside
    with left:
        top_left, top_right = st.columns(2)

        # Combined metrics bar
        with top_left:
            combined_df = pd.DataFrame({
                "Metrics": ["ATS Similarity", "AI Score", "Skill Match"],
                "Score": [
                    round(ats_score*100, 2),
                    round(ats_score*20, 2),
                    round((len(skills) / total_skills) * 100, 2)
                ]
            })
            fig_combined = px.bar(combined_df, x="Metrics", y="Score", text="Score")
            fig_combined.update_layout(margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_combined, use_container_width=True)
            st.markdown("**Summary:** Combined metrics show the overall fit. Higher is better.")

        # Keyword heatmap
        with top_right:
            if jd_keywords:
                freq_df = pd.DataFrame({
                    "Keyword": jd_keywords,
                    "Frequency": [len(re.findall(k, st.session_state.resume, re.IGNORECASE)) for k in jd_keywords]
                })
                heatmap = px.imshow([freq_df["Frequency"].tolist()], labels=dict(x="Keywords", y="Match Strength"))
                heatmap.update_layout(margin=dict(t=30, b=10, l=10, r=10))
                st.plotly_chart(heatmap, use_container_width=True)
                st.markdown("**Summary:** Heatmap shows which JD keywords are present in the resume.")

        # Next row: three compact charts
        row1, row2, row3 = st.columns(3)

        with row1:
            pie_df = pd.DataFrame({"Status": ["Matched", "Missing"], "Value": [len(skills), len(missing_keywords)]})
            fig_pie = px.pie(pie_df, values="Value", names="Status")
            fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown("**Summary:** Ratio of matched to missing keywords.")

        with row2:
            bar_df = pd.DataFrame({
                "Category": ["Matched Skills", "Missing Keywords"],
                "Count": [len(skills), len(missing_keywords)]
            })
            fig_bar = px.bar(bar_df, x="Category", y="Count", text="Count")
            fig_bar.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("**Summary:** Count comparison between matched and missing items.")

        with row3:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=ats_score * 100,
                title={'text': "ATS Similarity (%)"},
                gauge={'axis': {'range': [0, 100]},
                       'steps': [{'range': [0, 50], 'color': "#ff4b4b"},
                                 {'range': [50, 80], 'color': "#ffa64b"},
                                 {'range': [80, 100], 'color': "#00ADB5"}]}))
            gauge.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(gauge, use_container_width=True)
            st.markdown("**Summary:** Gauge indicates keyword alignment strength.")

        # Experience timeline
        exp_df = extract_experience_periods(st.session_state.resume)
        if not exp_df.empty:
            exp_df["Position"] = [f"Role {i+1}" for i in range(len(exp_df))]
            fig_tl = px.timeline(exp_df, x_start="Start", x_end="End", y="Position")
            fig_tl.update_layout(margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_tl, use_container_width=True)
            st.markdown("**Summary:** Extracted time ranges for candidate experience.")

    # Right: graphical report panel (compact insights)
    with right:
        st.markdown("### Graphical Summary")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("**Top matched skills:**", ", ".join([s.title() for s in skills]) or "None detected")
        st.write("**Top missing keywords from JD:**", ", ".join(missing_keywords) or "None")
        st.write("**ATS Similarity:**", f"{round(ats_score*100,2)}%")
        st.write("**AI Evaluation (approx):**", f"{round(ats_score*5,2)}/5")
        st.write("**Detected education:**", ", ".join(sorted(set(extract_resume_entities(st.session_state.resume)["EDUCATION"]))) or "N/A")
        st.markdown("</div>", unsafe_allow_html=True)

    # PDF report button
    if st.button("Generate PDF Report"):
        pdf_path = generate_pdf_report(report_text, ats_score, ats_score*5, skills, missing_keywords)
        try:
            with open(pdf_path, "rb") as f:
                st.download_button("Download AI Report", f, file_name="AI_Resume_Report.pdf")
        except Exception:
            st.error("Unable to open generated PDF file.")

    # Return to main menu
    st.markdown("---")
    if st.button("Return to Main Menu"):
        st.session_state.form_submitted = False
        st.rerun()
