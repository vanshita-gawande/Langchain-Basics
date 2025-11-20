# model_logic.py
import re   #Regular expressions for finding patterns (“X years”, words, etc.)
from typing import List, Dict  #NLP engine for tokenization, chunks, light skill extraction
import nltk
import spacy
from sentence_transformers import SentenceTransformer #Converts text → embedding vector
from sklearn.metrics.pairwise import cosine_similarity #Compare resume vs job description (for ATS score)
from transformers import pipeline  #Local model (FLAN-T5) for summarization
from langchain_community.llms import HuggingFacePipeline #Format prompts properly
from langchain_core.prompts import PromptTemplate #Wrap HF pipeline so LangChain can use it

# load resources (once : Models are heavy — loading them inside each request would make your app slow)
nlp = spacy.load("en_core_web_sm")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast local model
# Summarization / rewrite LLM (text2text)
pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=150, temperature=0.3)
llm = HuggingFacePipeline(pipeline=pipe)

# Basic skill vocabulary — can extend for your domain , To match skills from resume text.
COMMON_SKILLS = [
    "python","java","javascript","react","node.js","node","django","flask","fastapi",
    "postgresql","mysql","mongodb","aws","docker","kubernetes","git","rest","graphql",
    "nlp","machine learning","deep learning","pandas","numpy","tensorflow","pytorch"
]
# To match skills from resume text,Check if each skill from list is present,Add matched ones to found
def extract_skills(text: str, skill_list: List[str]=COMMON_SKILLS) -> List[str]:
    text_low = text.lower()
    found = []
    for skill in skill_list:
        # simple substring match; can be improved with fuzzy matching
        if skill.lower() in text_low:
            found.append(skill)
    # NER-based skill-ish extraction (tech nouns)
    doc = nlp(text)
    for ent in doc.noun_chunks:
        chunk = ent.text.strip().lower()
        if chunk not in found and len(chunk.split()) <= 3 and any(c.isalpha() for c in chunk):
            # optional filter
            pass
    return sorted(set(found))
#extract years of experience
def extract_years_experience(text: str) -> float:
    # naive: look for patterns like "X years"
    matches = re.findall(r'(\d+)\s+years', text.lower())
    if matches:
        nums = [int(m) for m in matches]
        return max(nums)
    return 0.0
# Compare: Resume text and Job description If similarity is high → resume fits job better.This is a MAJOR part of ATS systems today.
def semantic_similarity(a: str, b: str) -> float:
    emb_a = embed_model.encode([a])
    emb_b = embed_model.encode([b])
    sim = cosine_similarity(emb_a, emb_b)[0][0]
    return float(sim)

#More keyword matches → higher ATS score.
def keyword_match_score(resume_text: str, jd_text: str) -> float:
    # count how many job-keywords present in resume
    jd_tokens = set([w.lower() for w in re.findall(r'\w+', jd_text) if len(w) > 2])
    resume_tokens = set([w.lower() for w in re.findall(r'\w+', resume_text) if len(w) > 2])
    common = jd_tokens & resume_tokens
    if len(jd_tokens) == 0:
        return 0.0
    return len(common) / len(jd_tokens)

def compute_ats_score(resume_text: str, jd_text: str) -> Dict:
    # Combine: keyword match (50%) + semantic sim (40%) + experience factor (10%)
    kw_score = keyword_match_score(resume_text, jd_text)   # 0..1
    sem = semantic_similarity(resume_text, jd_text)        # -1..1 but usually 0..1
    sem_score = max(0.0, sem)  # ensure >=0
    years = extract_years_experience(resume_text)
    exp_score = min(years / 10.0, 1.0)  # saturate at 10 years

    total = 0.5 * kw_score + 0.4 * sem_score + 0.1 * exp_score #score
    return {
        "keyword_score": round(kw_score, 4),
        "semantic_score": round(sem_score, 4),
        "experience_score": round(exp_score, 4),
        "ats_score": round(total * 100, 2)  # scaled 0-100
    }

# Generate a short summary of the resume using LLM.This summary is used to create an ATS-friendly version.
# Summarize / rewrite resume summary to be ATS-friendly
summary_prompt = PromptTemplate.from_template(
    "Rewrite the following resume summary into a concise, ATS-friendly professional summary:\n\n{summary}"
)#LLM rewrites the summary to:be more powerful , use industry keywords , remove fluff
def rewrite_summary(summary_text: str) -> str:
    chain = summary_prompt | llm
    out = chain.invoke({"summary": summary_text})
    # chain.invoke returns a string
    return out

# Top-level analyze function This is what your FastAPI calls.
def analyze_resume(resume_text: str, job_description: str) -> Dict:
    skills = extract_skills(resume_text)
    years = extract_years_experience(resume_text)
    scores = compute_ats_score(resume_text, job_description)
    sem_sim = semantic_similarity(resume_text, job_description)
    # get a short summary of resume (use LLM)
    short_prompt = PromptTemplate.from_template("Summarize the following resume into 2-3 lines:\n\n{content}")
    chain = short_prompt | llm
    resume_summary = chain.invoke({"content": resume_text})
    # suggested summary rewrite
    rewritten = rewrite_summary(resume_summary)
    return {
        "skills": skills,
        "years_experience": years,
        "resume_summary": resume_summary,
        "rewritten_summary": rewritten,
        "scores": scores,
        "semantic_similarity": round(sem_sim, 4)
    }


# 🔥 WHAT IS model_logic.py?

# This file does all the actual AI work, including:

# ✔ Extracting skills from resume text
# ✔ Detecting years of experience
# ✔ Semantic similarity between resume & job description
# ✔ Keyword matching for ATS score
# ✔ Summarizing resume using local LLM
# ✔ Rewriting summary to ATS-friendly text
# ✔ Returning everything as final analysis