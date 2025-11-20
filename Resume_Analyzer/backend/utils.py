import os
from PyPDF2 import PdfReader

def save_upload(file, dest_folder="uploads"):
    os.makedirs(dest_folder, exist_ok=True) #creates folder upload exit ok means does not give error if folder exist , this is place to save resumes
    file_path = os.path.join(dest_folder, file.filename)#If resume is vanshita_resume.pdf, path becomes:uploads/vanshita_resume.pdf
    with open(file_path, "wb") as f: #wb for wriritng binary im for pdfs
        f.write(file.file.read())#read the uploaded files rwa bytes and write uploaded pdf to local folder
    return file_path

def extract_text_from_pdf(path):
    text_parts = []  #list to store text
    reader = PdfReader(path) #loads pdf
    for page in reader.pages: #if resume has pages then run loop through all pages
        try:
            text_parts.append(page.extract_text() or "") #returns text of that page
        except Exception:
            continue
    return "\n".join(text_parts) #join all into one big string perfect for analysis



# Purpose
# 👉 To save the uploaded resume (PDF) from the frontend into a folder on your system.

# Why?
# Because when a user uploads a PDF through FastAPI, it comes as a temporary streamed file.
# You must save it first so later functions (like PDF extraction) can read from disk.

# 🔥 PART 2 — extract_text_from_pdf()
# Purpose

# 👉 To read a PDF file and extract all text from every page.
# Why?
# The AI model cannot read PDF files directly.
# It needs raw text to analyze the resume.

# So we convert PDF → plain text.