# # model_logic.py
# import re   #Regular expressions for finding patterns (“X years”, words, etc.)
# from typing import List, Dict  #NLP engine for tokenization, chunks, light skill extraction
# import nltk
# import spacy
# from sentence_transformers import SentenceTransformer #Converts text → embedding vector
# from sklearn.metrics.pairwise import cosine_similarity #Compare resume vs job description (for ATS score)
# from transformers import pipeline  #Local model (FLAN-T5) for summarization
# from langchain_community.llms import HuggingFacePipeline #Format prompts properly
# from langchain_core.prompts import PromptTemplate #Wrap HF pipeline so LangChain can use it

# # load resources (once : Models are heavy — loading them inside each request would make your app slow)
# nlp = spacy.load("en_core_web_sm")
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast local model
# # Summarization / rewrite LLM (text2text)
# pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=150, temperature=0.3)
# llm = HuggingFacePipeline(pipeline=pipe)

# # Basic skill vocabulary — can extend for your domain , To match skills from resume text.
# COMMON_SKILLS = [
#     "python","java","javascript","react","node.js","node","django","flask","fastapi",
#     "postgresql","mysql","mongodb","aws","docker","kubernetes","git","rest","graphql",
#     "nlp","machine learning","deep learning","pandas","numpy","tensorflow","pytorch"
# ]
# # To match skills from resume text,Check if each skill from list is present,Add matched ones to found
# def extract_skills(text: str, skill_list: List[str]=COMMON_SKILLS) -> List[str]:
#     text_low = text.lower()
#     found = []
#     for skill in skill_list:
#         # simple substring match; can be improved with fuzzy matching
#         if skill.lower() in text_low:
#             found.append(skill)
#     # NER-based skill-ish extraction (tech nouns)
#     doc = nlp(text)
#     for ent in doc.noun_chunks:
#         chunk = ent.text.strip().lower()
#         if chunk not in found and len(chunk.split()) <= 3 and any(c.isalpha() for c in chunk):
#             # optional filter
#             pass
#     return sorted(set(found))
# #extract years of experience
# def extract_years_experience(text: str) -> float:
#     # naive: look for patterns like "X years"
#     matches = re.findall(r'(\d+)\s+years', text.lower())
#     if matches:
#         nums = [int(m) for m in matches]
#         return max(nums)
#     return 0.0
# # Compare: Resume text and Job description If similarity is high → resume fits job better.This is a MAJOR part of ATS systems today.
# def semantic_similarity(a: str, b: str) -> float:
#     emb_a = embed_model.encode([a])
#     emb_b = embed_model.encode([b])
#     sim = cosine_similarity(emb_a, emb_b)[0][0]
#     return float(sim)

# #More keyword matches → higher ATS score.
# def keyword_match_score(resume_text: str, jd_text: str) -> float:
#     # count how many job-keywords present in resume
#     jd_tokens = set([w.lower() for w in re.findall(r'\w+', jd_text) if len(w) > 2])
#     resume_tokens = set([w.lower() for w in re.findall(r'\w+', resume_text) if len(w) > 2])
#     common = jd_tokens & resume_tokens
#     if len(jd_tokens) == 0:
#         return 0.0
#     return len(common) / len(jd_tokens)

# def compute_ats_score(resume_text: str, jd_text: str) -> Dict:
#     # Combine: keyword match (50%) + semantic sim (40%) + experience factor (10%)
#     kw_score = keyword_match_score(resume_text, jd_text)   # 0..1
#     sem = semantic_similarity(resume_text, jd_text)        # -1..1 but usually 0..1
#     sem_score = max(0.0, sem)  # ensure >=0
#     years = extract_years_experience(resume_text)
#     exp_score = min(years / 10.0, 1.0)  # saturate at 10 years

#     total = 0.5 * kw_score + 0.4 * sem_score + 0.1 * exp_score #score
#     return {
#         "keyword_score": round(kw_score, 4),
#         "semantic_score": round(sem_score, 4),
#         "experience_score": round(exp_score, 4),
#         "ats_score": round(total * 100, 2)  # scaled 0-100
#     }

# # Generate a short summary of the resume using LLM.This summary is used to create an ATS-friendly version.
# # Summarize / rewrite resume summary to be ATS-friendly
# summary_prompt = PromptTemplate.from_template(
#     "Rewrite the following resume summary into a concise, ATS-friendly professional summary:\n\n{summary}"
# )#LLM rewrites the summary to:be more powerful , use industry keywords , remove fluff
# def rewrite_summary(summary_text: str) -> str:
#     chain = summary_prompt | llm
#     out = chain.invoke({"summary": summary_text})
#     # chain.invoke returns a string
#     return out

# # Top-level analyze function This is what your FastAPI calls.
# def analyze_resume(resume_text: str, job_description: str) -> Dict:
#     skills = extract_skills(resume_text)
#     years = extract_years_experience(resume_text)
#     scores = compute_ats_score(resume_text, job_description)
#     sem_sim = semantic_similarity(resume_text, job_description)
#     # get a short summary of resume (use LLM)
#     short_prompt = PromptTemplate.from_template("Summarize the following resume into 2-3 lines:\n\n{content}")
#     chain = short_prompt | llm
#     resume_summary = chain.invoke({"content": resume_text})
#     # suggested summary rewrite
#     rewritten = rewrite_summary(resume_summary)
#     return {
#         "skills": skills,
#         "years_experience": years,
#         "resume_summary": resume_summary,
#         "rewritten_summary": rewritten,
#         "scores": scores,
#         "semantic_similarity": round(sem_sim, 4)
#     }


# # 🔥 WHAT IS model_logic.py?

# # This file does all the actual AI work, including:

# # ✔ Extracting skills from resume text
# # ✔ Detecting years of experience
# # ✔ Semantic similarity between resume & job description
# # ✔ Keyword matching for ATS score
# # ✔ Summarizing resume using local LLM
# # ✔ Rewriting summary to ATS-friendly text
# # ✔ Returning everything as final analysis

# Refactored model_logic.py
# Clean, robust, and production-ready version of your original file.
# Fixes: proper HF pipeline output extraction, safer prompt handling,
# fewer duplicate chains, better error handling and clear return values.

import re
from typing import List, Dict, Any
import logging

import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Optional: if LangChain wrappers are available in your env you can use them,
# but the code below extracts generated text directly from HF pipeline outputs
# to avoid wrapping issues.
try:
    from langchain_community.llms import HuggingFacePipeline
    from langchain_core.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --------- Model load (do once) ---------
# Keep models small for local/dev; replace with heavier models in prod as needed.
_nlp = spacy.load("en_core_web_sm")
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
_pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=160, temperature=0.3)

# If LangChain wrapper is available, create it (optional)
_llm = HuggingFacePipeline(pipeline=_pipe) if LANGCHAIN_AVAILABLE else None

# --------- Skill vocabulary ---------
COMMON_SKILLS = [
    "python","java","javascript","react","node.js","node","django","flask","fastapi",
    "postgresql","mysql","mongodb","aws","docker","kubernetes","git","rest","graphql",
    "nlp","machine learning","deep learning","pandas","numpy","tensorflow","pytorch"
]

# --------- Helper utilities ---------

def _safe_extract_hf_text(hf_output: Any) -> str:
    """Extract plain text from HF pipeline or LangChain outputs.

    HF pipeline typically returns a list of dicts: [{"generated_text": "..."}]
    LangChain wrappers may return strings or other containers. This helper
    tries common shapes and falls back to str(out).
    """
    if hf_output is None:
        return ""

    # huggingface pipeline output
    if isinstance(hf_output, list) and len(hf_output) > 0:
        first = hf_output[0]
        if isinstance(first, dict) and "generated_text" in first:
            return first["generated_text"].strip()
        # some pipelines may return {'text': ...}
        if isinstance(first, dict) and "text" in first:
            return first["text"].strip()

    # LangChain may return plain string
    if isinstance(hf_output, str):
        return hf_output.strip()

    # If it's a dict with nested outputs
    if isinstance(hf_output, dict):
        for key in ("generated_text", "text", "output_text"):
            if key in hf_output:
                return str(hf_output[key]).strip()

    # fallback
    return str(hf_output).strip()


# --------- Text processing functions ---------

def extract_skills(text: str, skill_list: List[str] = COMMON_SKILLS) -> List[str]:
    text_low = (text or "").lower()
    found = set()
    for skill in skill_list:
        if skill.lower() in text_low:
            found.add(skill)

    # Optionally, use simple noun-chunk scanning to find more tech-noun-like phrases
    try:
        doc = _nlp(text or "")
        for chunk in doc.noun_chunks:
            c = chunk.text.strip().lower()
            # small heuristic: length 1-3 words and contains alpha
            if 0 < len(c.split()) <= 3 and any(ch.isalpha() for ch in c):
                # skip extremely common words
                if len(c) > 2 and c not in found:
                    # avoid adding sentences; keep short phrases only
                    if len(c.split()) <= 3:
                        # only add if looks like skill (contains letters and not just numbers)
                        found.add(c)
    except Exception:
        # if spaCy fails for any reason, continue with substring matches only
        logger.debug("spaCy noun-chunk extraction failed, continuing with substring matches")

    # canonicalize some tokens (optional)
    canonical = set()
    for s in found:
        # unify node/node.js
        if s == "node":
            canonical.add("node.js")
        else:
            canonical.add(s)

    return sorted(canonical)


def extract_years_experience(text: str) -> float:
    text = (text or "").lower()
    # matches like '3 years', '5+ years', '2 years of experience'
    matches = re.findall(r'(\d+)\+?\s+years', text)
    if matches:
        try:
            nums = [int(m) for m in matches]
            return float(max(nums))
        except Exception:
            return 0.0
    return 0.0


def semantic_similarity(a: str, b: str) -> float:
    try:
        emb_a = _embed_model.encode([a], convert_to_numpy=True)
        emb_b = _embed_model.encode([b], convert_to_numpy=True)
        sim = cosine_similarity(emb_a, emb_b)[0][0]
        return float(sim)
    except Exception as e:
        logger.exception("embedding similarity failed: %s", e)
        return 0.0


def keyword_match_score(resume_text: str, jd_text: str) -> float:
    jd_tokens = set([w.lower() for w in re.findall(r"\w+", jd_text or "") if len(w) > 2])
    resume_tokens = set([w.lower() for w in re.findall(r"\w+", resume_text or "") if len(w) > 2])
    if not jd_tokens:
        return 0.0
    common = jd_tokens & resume_tokens
    return len(common) / len(jd_tokens)


def compute_ats_score(resume_text: str, jd_text: str) -> Dict[str, float]:
    kw_score = keyword_match_score(resume_text, jd_text)
    sem = semantic_similarity(resume_text, jd_text)
    sem_score = max(0.0, sem)
    years = extract_years_experience(resume_text)
    exp_score = min(years / 10.0, 1.0)

    total = 0.5 * kw_score + 0.4 * sem_score + 0.1 * exp_score
    return {
        "keyword_score": round(kw_score, 4),
        "semantic_score": round(sem_score, 4),
        "experience_score": round(exp_score, 4),
        "ats_score": round(total * 100, 2),
    }


# --------- LLM helpers ---------

# Prompts (kept small & clear)
SUMMARIZE_PROMPT = (
    "Summarize the following resume into 2-3 concise lines focusing on achievements, role, and key skills:\n\n{content}"
)
REWRITE_PROMPT = (
    "Rewrite the following resume summary into a concise, ATS-friendly professional summary (1-2 lines). Use strong verbs and include relevant keywords:\n\n{summary}"
)


def _generate_text_via_pipe(prompt_text: str) -> str:
    """Run HF pipeline and safely extract generated text."""
    try:
        out = _pipe(prompt_text)
        return _safe_extract_hf_text(out)
    except Exception as e:
        logger.exception("HF pipeline failed: %s", e)
        return ""


def rewrite_summary(summary_text: str) -> str:
    if not summary_text:
        return ""
    prompt = REWRITE_PROMPT.format(summary=summary_text)
    # prefer direct pipeline; LangChain wrapper is optional
    return _generate_text_via_pipe(prompt)


def summarize_resume(resume_text: str) -> str:
    if not resume_text:
        return ""
    prompt = SUMMARIZE_PROMPT.format(content=resume_text)
    return _generate_text_via_pipe(prompt)


# --------- Top-level analyze function ---------

def analyze_resume(resume_text: str, job_description: str) -> Dict[str, Any]:
    """Main entry point used by FastAPI. Returns a JSON-serializable dict."""
    resume_text = resume_text or ""
    job_description = job_description or ""

    skills = extract_skills(resume_text)
    years = extract_years_experience(resume_text)
    scores = compute_ats_score(resume_text, job_description)
    sem_sim = semantic_similarity(resume_text, job_description)

    # Summarize and rewrite using HF pipeline
    try:
        resume_summary = summarize_resume(resume_text)
        rewritten = rewrite_summary(resume_summary)
    except Exception as e:
        logger.exception("LLM summary/rewrite failed: %s", e)
        resume_summary = ""
        rewritten = ""

    return {
        "skills": skills,
        "years_experience": years,
        "resume_summary": resume_summary,
        "rewritten_summary": rewritten,
        "scores": scores,
        "semantic_similarity": round(sem_sim, 4),
    }


# End of file.  
# Replace your old model_logic.py with this file and restart your FastAPI server.
# Notes:
# - This code prefers direct HF pipeline calls. If you want to use LangChain wrappers,
#   you can but ensure you extract text exactly as done by _safe_extract_hf_text().
# - Keep model loading at module import for dev convenience; in production you may
#   want lazy-loading or background loading to reduce startup time.
