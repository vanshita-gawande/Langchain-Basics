# app.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.utils import save_upload, extract_text_from_pdf
from backend.model_logic import analyze_resume

app = FastAPI()#creates fastapi app
#solved the connection issue as both runs on diff origin browser solved this issue
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    job_description: str
# Upload Resume Endpoint
@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    path = save_upload(file)
    text = extract_text_from_pdf(path)
    return {"path": path, "text_preview": text[:1000]}

# Analyze Resume Endpoint (main AI engine)
@app.post("/analyze")
async def analyze(path: str = Form(...), job_description: str = Form(...)):
    # path: path to saved resume (returned by upload-resume), or you can send raw text
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return {"error": "resume file not found"}
    resume_text = extract_text_from_pdf(str(p))
    result = analyze_resume(resume_text, job_description)
    return result

# simpler: accept direct text Useful for testing or using raw text directly instead of uploading PDF.
class AnalyzeByText(BaseModel):
    resume_text: str
    job_description: str

@app.post("/analyze-text")
def analyze_text(body: AnalyzeByText):
    return analyze_resume(body.resume_text, body.job_description)

# FULL SIMPLE SUMMARY

# Here’s how your backend works:

# Step 1 → Upload Resume
# Route: POST /upload-resume
# Frontend sends a PDF → backend:
# Saves file locally
# Extracts preview text

# Returns:
# {
#   "path": "uploads/resume.pdf",
#   "text_preview": "some resume text..."
# }
# Frontend stores the path.

# Step 2 → Analyze Resume
# Route: POST /analyze
# Frontend sends:
# path returned earlier
# job_description

# Backend:
# Reads PDF
# Extracts text
# Runs AI logic (skills, ATS score, summary, rewrite, etc.)
# Returns:

# {
#   "skills": [...],
#   "years_experience": ...,
#   "resume_summary": "...",
#   "rewritten_summary": "...",
#   "scores": { ... }
# }

# Step 3 → Display Results
# React UI shows the results.