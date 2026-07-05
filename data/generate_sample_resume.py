import fitz  # PyMuPDF
import os

def generate_pdf():
    # Define paths
    output_dir = os.path.join(os.path.dirname(__file__), "sample_resumes")
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "sample_resume.pdf")
    
    # Create empty PDF doc
    doc = fitz.open()
    page = doc.new_page()
    
    # Resume content text
    resume_content = """ALEX COOPER
Software Engineer & ML Developer | alex.cooper@example.com

SKILLS
Programming Languages: Python, Go, SQL, js (JavaScript), TypeScript
Frameworks & Libraries: FastAPI, PyTorch, scikit-learn, React, Streamlit, node, Express.js
Databases & Tools: PostgreSQL, SQLite, Docker, Git, AWS, mongo
Concepts: ml (Machine Learning), nlp (Natural Language Processing)

EXPERIENCE
Software Engineering Intern | TechCorp (Jan 2025 - Present)
- Engineered scalable microservices using FastAPI and SQLite, reducing latency by 20%.
- Integrated machine learning recommendation pipelines into web dashboards.
- Utilized PyMuPDF for automated document parsing and content processing.

PROJECTS
SkillHire AI Recommendation System (Personal Project)
- Developed an AI-powered job matching system with Streamlit frontend.
- Utilized sentence-transformers for semantic representation and matching.
- Implemented robust regex segmenters to extract text blocks.

EDUCATION
BS in Computer Science | University of California, Berkeley (2022 - 2026)
- Relevant Coursework: Machine Learning, Artificial Intelligence, Database Management Systems.

CERTIFICATIONS
AWS Certified Cloud Practitioner (2025)
TensorFlow Developer Certificate (2024)
"""
    
    # Insert text into page (margin-left: 50, margin-top: 50)
    page.insert_text((50, 50), resume_content, fontsize=11)
    
    # Save PDF
    doc.save(pdf_path)
    doc.close()
    print(f"Sample resume PDF generated successfully at: {pdf_path}")

if __name__ == "__main__":
    generate_pdf()
