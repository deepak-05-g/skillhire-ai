import logging
import requests
from typing import List, Dict, Any
from app.services.job_sources.normalizer import clean_html, split_description_and_requirements, parse_job_type

logger = logging.getLogger(__name__)

def fetch_greenhouse_jobs(board_token: str) -> List[Dict[str, Any]]:
    """
    Fetches job listings from the Greenhouse Public Job Board API for the given board token.
    Normalizes the response into the common job format.
    If the API call fails, logs the error and returns an empty list.
    """
    if not board_token:
        return []
        
    url = f"https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs"
    params = {"content": "true"}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        
        # Raise HTTP status error if code is 4xx or 5xx
        response.raise_for_status()
        
        data = response.json()
        jobs_list = data.get("jobs", [])
        
        normalized_jobs = []
        for job in jobs_list:
            title = job.get("title", "Unknown Title")
            raw_content = job.get("content", "")
            cleaned_content = clean_html(raw_content)
            
            # Split description and requirements
            desc, reqs = split_description_and_requirements(cleaned_content)
            
            # Determine job type
            job_type = parse_job_type(title, cleaned_content)
            
            # Normalize location
            location = job.get("location", {}).get("name", "Remote/Unknown")
            
            normalized_job = {
                "source": "Greenhouse",
                "company": board_token.title(),
                "title": title,
                "location": location,
                "description": desc,
                "requirements": reqs,
                "apply_url": job.get("absolute_url", ""),
                "job_type": job_type
            }
            normalized_jobs.append(normalized_job)
            
        return normalized_jobs
        
    except requests.exceptions.HTTPError as exc:
        logger.warning("Greenhouse API HTTP error for board '%s': %s", board_token, exc)
        return []
    except requests.exceptions.RequestException as exc:
        logger.warning("Greenhouse API network error for board '%s': %s", board_token, exc)
        return []
    except Exception as exc:
        logger.exception("Unexpected Greenhouse parsing error for board '%s'.", board_token)
        return []
