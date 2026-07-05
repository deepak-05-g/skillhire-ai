"""
Generate official Big Tech career search links from resume signals.

Uses only public, official career-site search URLs — no scraping or bot bypass.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

# Role keywords matched against resume text (first match wins by priority)
ROLE_RULES: List[tuple[str, str]] = [
    (r"\b(intern|internship)\b", "Software Engineering Intern"),
    (r"\b(machine learning|ml engineer|deep learning)\b", "Machine Learning Engineer"),
    (r"\b(data analyst|data analysis)\b", "Data Analyst"),
    (r"\b(data scientist)\b", "Data Scientist"),
    (r"\b(full[\s-]?stack)\b", "Full Stack Engineer"),
    (r"\b(frontend|front-end|front end)\b", "Frontend Engineer"),
    (r"\b(backend|back-end|back end)\b", "Backend Engineer"),
    (r"\b(devops|site reliability)\b", "DevOps Engineer"),
    (r"\b(cloud engineer|cloud computing)\b", "Cloud Engineer"),
    (r"\b(software engineer|software developer|sde)\b", "Software Engineer"),
]

HIGHLIGHT_SKILLS = {
    "python", "java", "javascript", "typescript", "react", "machine learning",
    "nlp", "sql", "docker", "aws", "azure", "kubernetes", "go", "c++",
}

LOCATION_HINTS = [
    "India", "United States", "USA", "Canada", "United Kingdom", "UK",
    "Germany", "Singapore", "Australia", "Remote",
    "Bengaluru", "Bangalore", "Hyderabad", "Mumbai", "Delhi", "Pune",
    "San Francisco", "Seattle", "New York", "Austin", "London",
]

OFFICIAL_COMPANIES = [
    {
        "company": "Google",
        "source_label": "Google Careers",
        "url_builder": "google",
    },
    {
        "company": "Microsoft",
        "source_label": "Microsoft Careers",
        "url_builder": "microsoft",
    },
    {
        "company": "Amazon",
        "source_label": "Amazon Jobs",
        "url_builder": "amazon",
    },
]


def infer_role(resume_text: str, resume_skills: List[str]) -> str:
    """Guess the most relevant job role from resume text and skills."""
    text_lower = resume_text.lower()

    for pattern, role in ROLE_RULES:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return role

    skills_lower = {s.lower() for s in resume_skills}
    if skills_lower & {"machine learning", "nlp", "tensorflow", "pytorch", "deep learning"}:
        return "Machine Learning Engineer"
    if skills_lower & {"react", "next.js", "html", "css", "javascript", "typescript"}:
        if skills_lower & {"python", "fastapi", "node.js", "django", "flask"}:
            return "Full Stack Engineer"
        return "Frontend Engineer"
    if skills_lower & {"python", "fastapi", "node.js", "django", "flask", "java"}:
        return "Software Engineer"

    return "Software Engineer"


def infer_location(resume_text: str, location: Optional[str] = None) -> str:
    """Use explicit location or detect a location hint from resume text."""
    if location and location.strip():
        return location.strip()

    text_lower = resume_text.lower()
    for hint in LOCATION_HINTS:
        if hint.lower() in text_lower:
            if hint.lower() == "usa":
                return "United States"
            if hint.lower() == "uk":
                return "United Kingdom"
            return hint

    return ""


def _pick_query_skills(resume_skills: List[str], max_count: int = 3) -> List[str]:
    """Select the most useful skills to include in a search query."""
    prioritized: List[str] = []
    other: List[str] = []

    for skill in resume_skills:
        if skill.lower() in HIGHLIGHT_SKILLS:
            prioritized.append(skill)
        else:
            other.append(skill)

    return (prioritized + other)[:max_count]


def build_search_query(role: str, resume_skills: List[str]) -> str:
    """Combine role and top skills into a single search string."""
    skill_part = _pick_query_skills(resume_skills)
    if skill_part:
        return f"{role} {' '.join(skill_part)}"
    return role


def build_google_careers_url(query: str, location: str = "") -> str:
    """Official Google Careers job search URL."""
    params: Dict[str, str] = {}
    if query.strip():
        params["q"] = query.strip()
    if location.strip():
        params["location"] = location.strip()

    base = "https://careers.google.com/jobs/results/"
    if not params:
        return base
    return f"{base}?{urlencode(params)}"


def build_microsoft_careers_url(query: str, location: str = "") -> str:
    """Official Microsoft Careers job search URL."""
    params: Dict[str, str] = {}
    if query.strip():
        params["q"] = query.strip()
    if location.strip():
        params["lc"] = location.strip()

    base = "https://jobs.careers.microsoft.com/global/en/search"
    if not params:
        return base
    return f"{base}?{urlencode(params)}"


def build_amazon_careers_url(query: str, location: str = "") -> str:
    """Official Amazon Jobs search URL."""
    params: Dict[str, str] = {}
    if query.strip():
        params["base_query"] = query.strip()
    if location.strip():
        params["loc_query"] = location.strip()

    base = "https://www.amazon.jobs/en/search"
    if not params:
        return base
    return f"{base}?{urlencode(params)}"


def _build_url(builder: str, query: str, location: str) -> str:
    builders = {
        "google": build_google_careers_url,
        "microsoft": build_microsoft_careers_url,
        "amazon": build_amazon_careers_url,
    }
    return builders[builder](query, location)


def _format_skill_phrase(skills: List[str]) -> str:
    """Build a readable skill list for explanations."""
    highlights = _pick_query_skills(skills, max_count=4)
    if not highlights:
        return "your profile"
    if len(highlights) == 1:
        return highlights[0]
    return ", ".join(highlights[:-1]) + f" and {highlights[-1]}"


def generate_official_search_sources(
    resume_text: str,
    resume_skills: List[str],
    location: Optional[str] = None,
    include_amazon: bool = True,
) -> List[Dict[str, Any]]:
    """
    Build official career search links for Google, Microsoft, and optionally Amazon.
    """
    role = infer_role(resume_text, resume_skills)
    resolved_location = infer_location(resume_text, location)
    query = build_search_query(role, resume_skills)
    skill_phrase = _format_skill_phrase(resume_skills)
    display_location = resolved_location or "Any location"

    companies = OFFICIAL_COMPANIES if include_amazon else OFFICIAL_COMPANIES[:2]
    results: List[Dict[str, Any]] = []

    for entry in companies:
        company = entry["company"]
        source_label = entry["source_label"]
        search_url = _build_url(entry["url_builder"], query, resolved_location)

        results.append(
            {
                "company": company,
                "title": f"Search {source_label} for {role} roles",
                "location": display_location,
                "source": f"{source_label} Official Search",
                "apply_url": search_url,
                "search_url": search_url,
                "match_score": None,
                "reason": (
                    f"Open this official {company} Careers search page to view current "
                    f"roles related to {skill_phrase} and {role}."
                ),
            }
        )

    return results
