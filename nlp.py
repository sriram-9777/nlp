import streamlit as st
import pandas as pd
import re
import nltk
from PyPDF2 import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
from nltk.corpus import stopwords


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

@st.cache_data
def load_data():
    df = pd.read_csv("glassdoor_jobs - 13.csv")
    df = df[['Job Title', 'Job Description']]
    df.dropna(inplace=True)
    df.drop_duplicates(subset=['Job Title'], inplace=True)
    df['clean_jd'] = df['Job Description'].apply(clean_text)
    return df

df = load_data()


tfidf = TfidfVectorizer(max_features=5000)
jd_vectors = tfidf.fit_transform(df['clean_jd'])


skills = [
    'python','machine learning','deep learning','sql',
    'nlp','aws','tensorflow','pytorch','excel','power bi'
]

def extract_skills(text):
    text = text.lower()
    return [skill for skill in skills if skill in text]


def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def ats_score(resume_text, job_text):
    resume_text = clean_text(resume_text)
    job_text = clean_text(job_text)
    vectors = tfidf.transform([resume_text, job_text])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(score * 100, 2)

def decision(score):
    if score >= 70:
        return "Selected "
    elif score >= 50:
        return "Review Required "
    else:
        return "Rejected "



st.set_page_config(page_title="AI ATS System", layout="centered")

st.title("🤖 AI-Based ATS Resume Screening System")

st.markdown("Upload your **resume** and get an **ATS prediction** instantly.")

uploaded_file = st.file_uploader(
    "Upload Resume (PDF or TXT)", type=["pdf", "txt"]
)


resume_text = ""
suggested_index = 0

if uploaded_file is not None:
    
    if uploaded_file.type == "application/pdf":
        resume_text = read_pdf(uploaded_file)
    else:
        resume_text = uploaded_file.read().decode("utf-8")
    
   
    clean_resume = clean_text(resume_text)
    resume_tfidf = tfidf.transform([clean_resume])
    
    
    similarities = cosine_similarity(resume_tfidf, jd_vectors).flatten()
    
    
    import numpy as np
    suggested_index = int(np.argmax(similarities))
    
    suggested_job = df.iloc[suggested_index]['Job Title']
    st.info(f"✨ **Resume Analysis:** We suggest **{suggested_job}** based on your profile.")

job_index = st.selectbox(
    "Select Job Title",
    df.index,
    format_func=lambda x: df.loc[x, 'Job Title'],
    index=suggested_index
)

job_text = df.loc[job_index, 'Job Description']

if uploaded_file is not None:
    if st.button(" Analyze Resume"):

        with st.spinner('Analyzing Resume...'):
            score = ats_score(resume_text, job_text)
            result = decision(score)

            resume_skills = extract_skills(resume_text)
            job_skills = extract_skills(job_text)
            missing_skills = list(set(job_skills) - set(resume_skills))
        
        
        st.subheader("ATS Analysis Result")
        
        
        col1, col2 = st.columns([1, 1])

        with col1:
            
            st.metric(label="Match Score", value=f"{score}%")
            st.progress(score / 100)
            
            
            if score >= 70:
                st.caption(" High Match")
            elif score >= 50:
                st.caption(" Medium Match")
            else:
                st.caption(" Low Match")

        with col2:
            st.markdown("### Decision")
            if result == "Selected ":
                st.success(f"**{result}**")
                st.markdown(" **One of the best candidates!**")
            elif result == "Review Required ":
                st.warning(f"**{result}**")
                st.markdown(" **Potential match, requires manual review.**")
            else:
                st.error(f"**{result}**")
                st.markdown(" **Profile does not match well.**")
            
            st.metric(label="Skills Matched", value=len(set(resume_skills).intersection(job_skills)), delta=f"{len(resume_skills)} found")

        
        st.divider()
        st.subheader("Skills Analysis")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("####  Skills Found")
            if resume_skills:
                
                st.markdown(" ".join([f"`{skill}`" for skill in resume_skills]))
            else:
                st.write("No specific skills detected.")

        with col_b:
            st.markdown("####  Missing Skills")
            if missing_skills:
                 
                 st.markdown(" ".join([f"`{skill}`" for skill in missing_skills]))
            else:
                st.success("Great! No critical skills missing from the target list.")