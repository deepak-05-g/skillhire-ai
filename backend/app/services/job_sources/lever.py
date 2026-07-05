import logging
import requests
from typing import List, Dict, Any
from app.services.job_sources.normalizer import clean_html, split_description_and_requirements, parse_job_type

logger = logging.getLogger(__name__)

def fetch_lever_jobs(company_id: str) -> List[Dict[str, Any]]:
    """
    Fetches job listings from the Lever Postings API for the given company handle.
    Normalizes the response into the common job format.
    If the API call fails, logs the error and returns an empty list.
    """
    if not company_id:
        return []
        
    url = f"https://api.lever.co/v0/postings/{company_id}"
    
    try:
        response = requests.get(url, timeout=10)
        
        # Raise HTTP status error if code is 4xx or 5xx
        response.raise_for_status()
        
        data = response.json()
        if not isinstance(data, list):
            logger.warning(
                "Lever API returned invalid data for company '%s' (expected list).",
                company_id,
            )
            return []
            
        normalized_jobs = []
        for job in data:
            title = job.get("title", "Unknown Title")
            hosted_url = job.get("hostedUrl", "")
            
            # Extract location and commitment
            categories = job.get("categories", {})
            location = categories.get("location", "Remote/Unknown")
            commitment = categories.get("commitment", "Unknown")
            
            # Construct description and extract requirements
            raw_desc = job.get("descriptionHtml", "")
            cleaned_desc = clean_html(raw_desc)
            
            # Lever has a structured lists array e.g., qualifications, requirements, etc.
            reqs_list = []
            lists = job.get("lists", [])
            for lst in lists:
                lst_title = lst.get("text", "")
                lst_content = lst.get("content", "")
                
                # Check if this list contains requirements or qualifications
                if any(k in lst_title.lower() for k in ["requirement", "qualification", "what you", "look for", "need", "skill"]):
                    cleaned_lst_content = clean_html(lst_content)
                    reqs_list.append(f"{lst_title}:\n{cleaned_lst_content}")
                else:
                    # Append other lists (e.g. Responsibilities) to the description
                    cleaned_lst_content = clean_html(lst_content)
                    cleaned_desc += f"\n\n{lst_title}\n{cleaned_lst_content}"
            
            # If we found requirements in lists, combine them
            if reqs_list:
                requirements = "\n\n".join(reqs_list)
            else:
                # Fallback to splitting standard description if lists are empty/non-matching
                cleaned_desc, requirements = split_description_and_requirements(cleaned_desc)
                
            # Determine job type
            job_type = parse_job_type(title, f"{commitment} {cleaned_desc} {requirements}")
            if commitment.lower() == "full-time" or commitment.lower() == "full time":
                job_type = "Full-time"
            elif any(k in commitment.lower() for k in ["intern", "coop"]):
                job_type = "Internship"
                
            normalized_job = {
                "source": "Lever",
                "company": company_id.title(),
                "title": title,
                "location": location,
                "description": cleaned_desc,
                "requirements": requirements,
                "apply_url": hosted_url,
                "job_type": job_type
            }
            normalized_jobs.append(normalized_job)
            
        return normalized_jobs
        
    except requests.exceptions.HTTPError as exc:
        logger.warning("Lever API HTTP error for company '%s': %s", company_id, exc)
        return []
    except requests.exceptions.RequestException as exc:
        logger.warning("Lever API network error for company '%s': %s", company_id, exc)
        return []
    except Exception as exc:
        logger.exception("Unexpected Lever parsing error for company '%s'.", company_id)
        return []
