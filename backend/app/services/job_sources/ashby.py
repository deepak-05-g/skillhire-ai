import logging
import requests
from typing import List, Dict, Any
from app.services.job_sources.normalizer import split_description_and_requirements, parse_job_type

logger = logging.getLogger(__name__)

def fetch_ashby_jobs(company_id: str) -> List[Dict[str, Any]]:
    """
    Fetches job listings from the Ashby Public Posting API for the given organization/company handle.
    Normalizes the response into the common job format.
    If the API call fails, logs the error and returns an empty list.
    """
    if not company_id:
        return []
        
    url = f"https://api.ashbyhq.com/posting-api/job-board/{company_id}"
    
    try:
        response = requests.get(url, timeout=10)
        
        # Raise HTTP status error if code is 4xx or 5xx
        response.raise_for_status()
        
        data = response.json()
        jobs_list = data.get("jobs", [])
        
        normalized_jobs = []
        for job in jobs_list:
            title = job.get("title", "Unknown Title")
            location = job.get("location", "Remote/Unknown")
            employment_type = job.get("employmentType", "Unknown")
            job_url = job.get("jobUrl", "")
            
            # Ashby gives plain text description directly in descriptionPlain
            description_plain = job.get("descriptionPlain", "")
            
            # Split description and requirements
            desc, reqs = split_description_and_requirements(description_plain)
            
            # Determine job type from employmentType or text
            job_type = "Unknown"
            if employment_type.lower() in ["fulltime", "full-time", "full_time"]:
                job_type = "Full-time"
            elif employment_type.lower() in ["intern", "internship", "coop", "co-op"]:
                job_type = "Internship"
            else:
                # Fallback to checking title/description keywords
                job_type = parse_job_type(title, description_plain)
                
            normalized_job = {
                "source": "Ashby",
                "company": company_id.title(),
                "title": title,
                "location": location,
                "description": desc,
                "requirements": reqs,
                "apply_url": job_url,
                "job_type": job_type
            }
            normalized_jobs.append(normalized_job)
            
        return normalized_jobs
        
    except requests.exceptions.HTTPError as exc:
        logger.warning("Ashby API HTTP error for company '%s': %s", company_id, exc)
        return []
    except requests.exceptions.RequestException as exc:
        logger.warning("Ashby API network error for company '%s': %s", company_id, exc)
        return []
    except Exception as exc:
        logger.exception("Unexpected Ashby parsing error for company '%s'.", company_id)
        return []
