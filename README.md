# SkillHire AI 🚀

SkillHire AI is an intelligent, resume-based job recommendation system that simplifies the job search process. By simply uploading a resume, users receive highly personalized, ranked job matches tailored to their specific skills and experience.

## Live Render URLs

- Frontend app: https://skillhire-frontend.onrender.com
- Backend API docs: https://skillhire-backend.onrender.com/docs
- Backend health check: https://skillhire-backend.onrender.com/health

Render free-tier services may take a minute to wake up after inactivity.

## 🎯 Problem Statement

The modern job search is broken. Candidates spend hours sifting through thousands of irrelevant job postings, manually parsing requirements, and second-guessing if their skills align with the role. SkillHire AI solves this by inverting the process: **bring your resume, and let the AI find the jobs that fit you.**

## ✨ Features

- **Automated Resume Parsing**: Extracts skills, projects, education, and experience from PDF resumes using PyMuPDF and NLP techniques.
- **Smart Skill Extraction**: Maps free-text resume data into a normalized skill taxonomy.
- **Semantic Job Matching**: Uses Sentence-Transformers to understand the contextual and semantic overlap between your resume and a job description.
- **Job Fit Classification**: A trained Random Forest model predicts if a job is a "High Fit", "Medium Fit", or "Low Fit" based on multiple overlapping factors.
- **Interactive UI**: Built with Streamlit, offering a clean, responsive, and intuitive user experience.

## 🛠️ Tech Stack

- **Frontend**: Streamlit, Pandas, Matplotlib, Requests
- **Backend**: Python, FastAPI, PostgreSQL/SQLite, SQLAlchemy, Uvicorn
- **Machine Learning & NLP**: Scikit-learn, Sentence-Transformers (all-MiniLM-L6-v2), PyMuPDF (fitz)
- **Deployment**: Docker, Docker Compose, Render

## 🏗️ Architecture Diagram

```text
+-------------------+       REST API        +-------------------+
|                   |  Upload Resume (PDF)  |                   |
|   Streamlit UI    | --------------------> |    FastAPI        |
|   (Frontend)      |                       |    (Backend)      |
|                   | <-------------------- |                   |
+-------------------+    Ranked Job Matches +---------+---------+
                                                      |
                                                      |
          +-----------------------+-------------------+-----------------------+
          |                       |                                           |
          v                       v                                           v
+-------------------+   +--------------------+                      +-------------------+
|   Resume Parser   |   |   Job Fetcher      |                      |  Matching Engine  |
| - PyMuPDF Text    |   | - Public APIs      |                      | - Skill Overlap   |
| - Regex Segment   |   | - Job Boards       |                      | - Semantic Sim.   |
+-------------------+   +--------------------+                      +-------------------+
                                                                              |
                                                                              v
                                                                    +-------------------+
                                                                    |  Fit Classifier   |
                                                                    | - Random Forest   |
                                                                    +-------------------+
```

## 🧠 ML Approach & Matching Score

### How the Matching Score is Calculated
The final match score is a weighted composite of three primary factors:
1. **Skill Match (45%)**: Computes the exact overlap between the canonical skills extracted from your resume and the required skills found in the job description.
2. **Semantic Similarity (45%)**: Uses the `all-MiniLM-L6-v2` Sentence-Transformer model to generate high-dimensional embeddings of both the resume and the job text, computing the cosine similarity between them to capture contextual relevance beyond mere keyword matching.
3. **Role/Title Bonus (10%)**: Applies a small heuristic bonus if your past project experience and skills directly align with the core job title (e.g., "Software Engineer", "Data Analyst").

### Fit Classification Model
A supervised Random Forest classifier predicts the overall fit category (Low, Medium, High). Currently, this is trained on a synthetic dataset of resume-job pairs built around typical tech roles, utilizing the matching metrics as feature inputs.

## 📡 APIs & Sources Used

The application retrieves job listings from:
- Greenhouse (via public job board APIs)
- Lever (via public job board APIs)
- Ashby (via public job board APIs)
- Direct official career search links (No direct scraping of Google/Microsoft portals is performed).

## ⚠️ Limitations

- **No Big Tech Direct Scraping**: Direct scraping of heavily gated career sites like Google or Microsoft is intentionally omitted to respect their terms of service.
- **Job Sources**: The current MVP relies on public job board APIs (Greenhouse, Lever, Ashby) and official career search links.
- **Classifier Training Data**: The "Fit Classifier" currently uses a small, synthetically generated dataset for its MVP release. It can be significantly improved by fine-tuning with real-world application outcomes and user feedback data.

## 🚀 Setup Instructions

### Prerequisites
- Docker and Docker Compose
- Or, Python 3.10+ (for manual setup)

### Running with Docker (Recommended)

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd skillhire-ai
   ```

2. **Set up Environment Variables:**
   ```bash
   cp .env.example .env
   ```

3. **Build and start the containers:**
   ```bash
   docker-compose up --build
   ```

4. **Access the application:**
   - Frontend UI: http://localhost:8501
   - Backend API Docs: http://localhost:8000/docs

Docker Compose starts a PostgreSQL database automatically:

```text
postgresql+psycopg://skillhire:skillhire@localhost:5432/skillhire_ai
```

Inside Docker, the backend uses:

```text
postgresql+psycopg://skillhire:skillhire@db:5432/skillhire_ai
```

The backend creates the current SQLAlchemy tables on startup.

### Manual Local Setup

1. **Start the Backend:**
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn app.main:app --reload --port 8000
   ```

2. **Start the Frontend:**
   ```bash
   cd frontend
   pip install -r requirements.txt
   streamlit run app.py
   ```

### Database Configuration

By default, a manual backend run uses a local SQLite file at `backend/skillhire_ai.db`.
For a proper server database, set `DATABASE_URL` before starting the backend:

```bash
DATABASE_URL=postgresql+psycopg://username:password@host:5432/database_name
```

If your cloud provider gives a URL like `postgres://...` or `postgresql://...`, the app normalizes it to the installed `psycopg` driver automatically.

## 📸 Screenshots

![Dashboard Placeholder](https://via.placeholder.com/800x400?text=SkillHire+AI+Dashboard)
*(Replace with actual screenshot of the Streamlit dashboard)*

![Match Results Placeholder](https://via.placeholder.com/800x400?text=Ranked+Job+Matches)
*(Replace with actual screenshot of the job matching results)*

## 🔮 Future Improvements

- Integrate real-world application tracking data to improve the Random Forest Fit Classifier.
- Add OAuth integration (e.g., LinkedIn, GitHub) for automatic profile importing.
- Implement an automated web crawler with proxy rotation for broader job board aggregation.
- Deploy the application to a cloud provider (AWS, GCP, or Azure) using Kubernetes.

## 📝 Resume Bullet Points

If you are a developer looking to add this to your resume, consider these bullet points:
- Architected and deployed an AI-driven job recommendation system using **FastAPI** and **Streamlit**, enabling users to receive highly personalized job matches based on PDF resume parsing.
- Integrated a hybrid ranking engine leveraging **Sentence-Transformers** for semantic similarity and canonical skill taxonomy mapping, increasing job relevance accuracy.
- Developed a **Random Forest** classification model using **Scikit-learn** to categorize candidate-job fit, processing multi-dimensional text features.
- Containerized the full-stack application with **Docker** and **Docker Compose**, streamlining the deployment lifecycle and ensuring consistent cross-environment execution.
