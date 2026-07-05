from app.services.job_sources.greenhouse import fetch_greenhouse_jobs
from app.services.job_sources.lever import fetch_lever_jobs
from app.services.job_sources.ashby import fetch_ashby_jobs
from typing import List, Dict, Any

def fetch_jobs(source: str, company: str) -> List[Dict[str, Any]]:
    """
    Unified entry point to fetch and normalize job listings from a specified source
    (greenhouse, lever, or ashby) and company board handle.
    """
    source_lower = source.lower().strip()
    company_clean = company.strip()
    
    if source_lower == "greenhouse":
        return fetch_greenhouse_jobs(company_clean)
    elif source_lower == "lever":
        return fetch_lever_jobs(company_clean)
    elif source_lower == "ashby":
        return fetch_ashby_jobs(company_clean)
    else:
        raise ValueError(f"Unsupported job board source: '{source}'. Choose 'greenhouse', 'lever', or 'ashby'.")
