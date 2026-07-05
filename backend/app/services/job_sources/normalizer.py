"""Shared helpers for normalizing job-board HTML/text into a common schema."""

from __future__ import annotations

import html
import logging
import re
from typing import Tuple

logger = logging.getLogger(__name__)

# Normalized job keys returned by Greenhouse, Lever, and Ashby connectors.
COMMON_JOB_FIELDS: tuple[str, ...] = (
    "source",
    "company",
    "title",
    "location",
    "description",
    "requirements",
    "apply_url",
    "job_type",
)


def clean_html(html_text: str) -> str:
    """
    Strip HTML tags and decode entities from a job description.

    Args:
        html_text: Raw HTML from a job board API.

    Returns:
        Plain-text job description.
    """
    if not html_text:
        return ""

    text = html_text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = re.sub(r"</?(?:p|div|li|h[1-6]|ul|ol)>", "\n", text)
    text = re.compile(r"<[^>]+>").sub("", text)
    text = html.unescape(text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_description_and_requirements(plain_text: str) -> Tuple[str, str]:
    """
    Split plain job text into description and requirements sections.

    Args:
        plain_text: Cleaned job posting text.

    Returns:
        Tuple of (description, requirements).
    """
    if not plain_text:
        return "", ""

    req_patterns = [
        r"(?:\n\s*|^|\.\s+)(?P<header>requirements|qualifications|what\s+you(?:\'ll|will)\s+need|what\s+we(?:\'re|are)\s+looking\s+for|skills\s+(?:required|needed)|experience\s+required|basic\s+qualifications|what\s+you\s+bring)\b"
    ]

    for pattern in req_patterns:
        match = re.search(pattern, plain_text, re.IGNORECASE)
        if match:
            header_start = match.start("header")
            description = plain_text[:header_start].strip()
            requirements = plain_text[header_start:].strip()
            header_text = match.group("header")
            requirements_content = requirements[len(header_text):].strip()
            requirements_content = re.sub(r"^[:\-]?\s*", "", requirements_content).strip()
            logger.debug("Split job text at requirements header '%s'.", header_text)
            return description, requirements_content

    return plain_text, "Refer to the main description."


def parse_job_type(title: str, text: str) -> str:
    """
    Infer job type from title and body keywords.

    Returns:
        One of ``Internship``, ``Full-time``, or ``Unknown``.
    """
    combined = f"{title} {text}".lower()

    if any(keyword in combined for keyword in ("intern", "internship", "co-op", "coop")):
        return "Internship"
    if any(keyword in combined for keyword in ("fulltime", "full-time", "full time", "ft")):
        return "Full-time"
    return "Unknown"
