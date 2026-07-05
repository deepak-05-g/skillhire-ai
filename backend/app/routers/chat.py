import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatMessage(BaseModel):
    role: str = Field(..., description="Either 'user' or 'assistant'/'model'")
    content: str = Field(..., description="Message text content")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Conversation history")
    resume_text: str = Field(default="", description="Parsed resume text for richer career guidance")
    resume_skills: List[str] = Field(default_factory=list, description="Extracted resume skills")
    missing_skills: List[str] = Field(default_factory=list, description="Extracted missing skills/skill gaps")
    job_recommendations: List[dict] = Field(default_factory=list, description="Top recommended jobs list")
    career_goal: Optional[str] = Field(default=None, description="User's target role, domain, or career goal")


class ChatResponse(BaseModel):
    response: str


def _truncate_text(text: str, max_chars: int = 4000) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def _infer_role_suggestions(skills: List[str]) -> List[str]:
    skills_lower = {skill.lower() for skill in skills}
    suggestions = []

    if {"python", "fastapi", "sql", "postgresql"} & skills_lower:
        suggestions.append("Backend Engineer")
    if {"react", "javascript", "html", "css"} & skills_lower:
        suggestions.append("Frontend Developer")
    if {"machine learning", "scikit-learn", "pandas", "numpy"} & skills_lower:
        suggestions.append("Machine Learning Intern")
    if {"node.js", "express.js", "mongodb"} & skills_lower:
        suggestions.append("Full Stack Developer")
    if {"docker", "kubernetes", "aws", "gcp", "azure"} & skills_lower:
        suggestions.append("Cloud or DevOps Intern")

    return suggestions[:4] or [
        "Software Engineer Intern",
        "Backend Developer",
        "Full Stack Developer",
    ]


def _job_summary(job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": job.get("title", "Unknown role"),
        "company": job.get("company", "Unknown company"),
        "location": job.get("location", ""),
        "match_score": job.get("match_score", 0),
        "fit_label": job.get("fit_label", ""),
        "matched_skills": job.get("matched_skills", [])[:8],
        "missing_skills": job.get("missing_skills", [])[:8],
        "reason": job.get("reason", ""),
    }


def generate_fallback_response(request: ChatRequest) -> str:
    """
    Generate helpful rule-based guidance when Gemini is missing or unavailable.
    """
    user_query = request.messages[-1].content.lower() if request.messages else ""
    inferred_roles = _infer_role_suggestions(request.resume_skills)
    current_skills = {skill.lower() for skill in request.resume_skills}

    response = (
        "**SkillHire AI Advisor (Local Fallback Mode)**\n\n"
        "*Gemini is unavailable right now, so I am using the local advisor logic.*\n\n"
    )

    if any(k in user_query for k in ["career", "path", "type", "kind", "roles", "role", "find"]):
        response += "### Suggested Job Directions\n"
        if request.career_goal:
            response += f"Your stated goal is **{request.career_goal}**. Keep that as the main search theme.\n\n"
        response += "Based on the resume skills available, start with these role types:\n"
        for role in inferred_roles:
            response += f"- **{role}**\n"
        if request.resume_skills:
            skills = ", ".join(f"`{s}`" for s in request.resume_skills[:10])
            response += f"\nStrong keywords already present: {skills}.\n"
        response += "\nNext step: run job matching, then ask which exact roles are strongest from your ranked results."

    elif any(k in user_query for k in ["skill", "improve", "learn", "gap", "study", "roadmap"]):
        response += "### Skill Development Roadmap\n"
        if request.missing_skills:
            gaps = ", ".join(f"`{s}`" for s in request.missing_skills[:5])
            response += f"Based on your target jobs, your key missing skills/gaps are: {gaps}.\n\n"
            response += "Here is a step-by-step roadmap to acquire them:\n"
            for index, skill in enumerate(request.missing_skills[:3], 1):
                response += f"{index}. **{skill}**:\n"
                response += "   - *How to learn:* Start with official docs and one focused tutorial.\n"
                response += f"   - *Practice project:* Build a small portfolio project using `{skill}`.\n"
                response += "   - *Resume update:* Add one measurable bullet about the project.\n"
        else:
            response += "No matched-job gaps are loaded yet.\n\n"
            response += "Useful next skills to consider:\n"
            for skill in ["Docker", "PostgreSQL", "REST API design", "cloud deployment", "testing with pytest"]:
                if skill.lower() not in current_skills:
                    response += f"- **{skill}**\n"

    elif any(k in user_query for k in ["job", "apply", "recommend", "match", "company", "role"]):
        response += "### Personalized Job Search Advice\n"
        if request.job_recommendations:
            response += "Here are the top matches from your loaded listings:\n\n"
            for job in [_job_summary(item) for item in request.job_recommendations[:3]]:
                title = job.get("title", "Job Title")
                company = job.get("company", "Company")
                score = job.get("match_score", 0)
                missing = job.get("missing_skills", [])
                response += f"- **{title}** at **{company}**\n"
                response += f"  - *Match Score:* `{score}%` | *Fit:* {job.get('fit_label') or 'Active Match'}\n"
                if missing:
                    response += f"  - *Missing skills to add:* {', '.join(f'`{m}`' for m in missing[:3])}\n"
                response += "\n"
            response += "Before applying, tailor your summary and project bullets to the top requirements."
        else:
            response += "No ranked jobs are loaded yet. Based on your resume, begin with these searches:\n"
            for role in inferred_roles:
                response += f"- {role}\n"
            response += "\nThen load jobs and run matching so I can compare exact roles and skill gaps."

    elif any(k in user_query for k in ["resume", "tip", "profile", "cv", "portfolio"]):
        response += "### Resume Optimization Suggestions\n"
        if request.resume_skills:
            response += f"Your resume has **{len(request.resume_skills)}** skills identified.\n\n"
            response += "**General tips:**\n"
            response += "- Show impact with measurable project or internship outcomes.\n"
            response += "- Add GitHub, live demos, or project links where possible.\n"
            response += "- Repeat important job keywords naturally in project bullets, not only in the skills list.\n"
        else:
            response += "Upload and parse your resume first so I can give specific profile feedback."

    else:
        response += "### Hello! I am your SkillHire AI Career Advisor.\n"
        response += "I can help with job direction, skill gaps, resume updates, and application strategy.\n\n"
        if request.resume_skills:
            skills = ", ".join(f"`{s}`" for s in request.resume_skills[:8])
            response += f"**Skills loaded:** {skills}...\n"
            response += f"**Likely role directions:** {', '.join(inferred_roles)}\n"
        if request.missing_skills:
            gaps = ", ".join(f"`{s}`" for s in request.missing_skills[:5])
            response += f"**Skill gaps:** {gaps}\n"
        if request.job_recommendations:
            response += f"**Job matches:** {len(request.job_recommendations)} roles loaded\n"

        response += "\nTry asking me questions like:\n"
        response += "- *What type of jobs can I target with this resume?*\n"
        response += "- *What skills should I learn to bridge my gaps?*\n"
        response += "- *Which of my recommended jobs is the best fit?*\n"
        response += "- *How can I improve my resume?*\n"

    return response


@router.post("", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Conversational career advisor endpoint backed by Gemini, with a local fallback.
    """
    if not request.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chat history cannot be empty.",
        )

    api_key = settings.GEMINI_API_KEY
    if not api_key:
        return ChatResponse(response=generate_fallback_response(request))

    system_instruction = (
        "You are SkillHire AI Career Advisor, a practical resume-aware job search assistant.\n"
        "Help candidates understand what jobs they can target after uploading a resume, "
        "which skills to improve, how to prioritize job matches, and how to strengthen their resume.\n"
        "Be specific and actionable. Prefer concise sections with bullets. "
        "Do not invent companies or job postings not present in the provided context.\n"
        "When giving skill advice, separate must-have gaps from nice-to-have improvements "
        "and include one portfolio project idea when useful.\n\n"
    )

    if request.career_goal:
        system_instruction += f"Candidate career goal or target role: {request.career_goal}\n"

    if request.resume_text or request.resume_skills or request.missing_skills or request.job_recommendations:
        system_instruction += "Here is the candidate's current application context:\n"
        if request.resume_text:
            system_instruction += f"- Parsed resume excerpt: {_truncate_text(request.resume_text)}\n"
        if request.resume_skills:
            system_instruction += f"- Candidate's current skills: {', '.join(request.resume_skills)}\n"
        if request.missing_skills:
            system_instruction += (
                "- Candidate's skill gaps frequently missing from matched jobs: "
                f"{', '.join(request.missing_skills)}\n"
            )
        if request.job_recommendations:
            jobs_summary = []
            for job in [_job_summary(item) for item in request.job_recommendations[:5]]:
                jobs_summary.append(
                    "  * "
                    f"{job['title']} at {job['company']} ({job['location']}) - "
                    f"Match Score: {job['match_score']}%, "
                    f"Matched: {', '.join(job['matched_skills']) or 'not listed'}, "
                    f"Missing: {', '.join(job['missing_skills']) or 'none'}"
                )
            system_instruction += "- Top job matches:\n" + "\n".join(jobs_summary) + "\n"

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)

        contents = []
        for message in request.messages:
            role = "user" if message.role == "user" else "model"
            contents.append({"role": role, "parts": [message.content]})

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=system_instruction,
        )

        response = model.generate_content(contents)
        if response.text:
            return ChatResponse(response=response.text)
        raise ValueError("Empty response received from Gemini.")

    except Exception as exc:
        logger.error("Failed to generate response using Gemini: %s. Falling back to local responder.", exc)
        return ChatResponse(response=generate_fallback_response(request))
