"""Resume PDF parsing: text extraction, cleaning, and section segmentation."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class ResumeParsingError(Exception):
    """Raised when a resume PDF cannot be read or parsed."""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract raw text from PDF bytes using PyMuPDF.

    Args:
        file_bytes: Raw PDF file content.

    Returns:
        Extracted text from all pages.

    Raises:
        ResumeParsingError: If the PDF is invalid, empty, or unreadable.
    """
    if not file_bytes:
        raise ResumeParsingError("Cannot parse an empty PDF file.")

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        if len(doc) == 0:
            raise ResumeParsingError("The uploaded PDF has 0 pages.")

        text_parts = [doc.load_page(page_num).get_text() for page_num in range(len(doc))]
        raw_text = "".join(text_parts)

        if not raw_text.strip():
            raise ResumeParsingError(
                "The PDF appears empty or contains only non-extractable scanned images."
            )

        logger.info("Extracted text from PDF (%d pages, %d chars).", len(doc), len(raw_text))
        return raw_text

    except fitz.FileDataError as exc:
        logger.warning("Invalid PDF structure: %s", exc)
        raise ResumeParsingError(f"Invalid PDF file structure: {exc}") from exc
    except ResumeParsingError:
        raise
    except Exception as exc:
        logger.exception("Unexpected PDF parsing failure.")
        raise ResumeParsingError(f"Failed to read PDF file: {exc}") from exc


def clean_text(text: str) -> str:
    """
    Normalize whitespace and remove decorative separator lines.

    Args:
        text: Raw extracted resume text.

    Returns:
        Cleaned plain text.
    """
    if not text:
        return ""

    text = re.sub(r"\t+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"([-=_*~])\1{2,}", "", text)
    return text.strip()


def segment_sections(text: str) -> Dict[str, str]:
    """
    Split resume text into standard sections using heading patterns.

    Args:
        text: Cleaned resume text.

    Returns:
        Dict with keys: skills, projects, education, experience, certifications.
    """
    sections: Dict[str, str] = {
        "skills": "",
        "projects": "",
        "education": "",
        "experience": "",
        "certifications": "",
    }

    section_patterns = {
        "skills": r"^(?:technical\s+)?skills(?:\s+&\s+tools|\s+and\s+technologies)?\b|^\bcore\s+competencies\b|^\btechnologies\b|^\bprogramming\s+languages\b",
        "projects": r"^\b(?:academic\s+|personal\s+|key\s+|selected\s+)?projects\b",
        "education": r"^\b(?:academic\s+)?education\b|^\bacademic\s+(?:background|profile|credentials)\b|^\beducational\s+background\b",
        "experience": r"^\b(?:work\s+|professional\s+|employment\s+|relevant\s+)?experience\b|^\bemployment\s+history\b|^\bwork\s+history\b",
        "certifications": r"^\b(?:licenses\s+&\s+)?certifications\b|^\bcredentials\b|^\bprofessional\s+certifications\b|^\bawards\s+&\s+certifications\b",
    }

    matches: list[tuple[int, int, str]] = []
    for sec_name, pattern in section_patterns.items():
        for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            start_idx = match.start()
            line_end = text.find("\n", start_idx)
            if line_end == -1:
                line_end = len(text)
            line_text = text[start_idx:line_end].strip()
            if len(line_text) < 40:
                matches.append((start_idx, line_end, sec_name))

    matches.sort(key=lambda item: item[0])

    for i, (start, end, sec_name) in enumerate(matches):
        next_start = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        content = text[end:next_start].strip()
        if sections[sec_name]:
            sections[sec_name] += "\n\n" + content
        else:
            sections[sec_name] = content

    logger.debug(
        "Segmented resume into sections: %s",
        {key: bool(value.strip()) for key, value in sections.items()},
    )
    return sections


def parse_resume(file_bytes: bytes) -> Dict[str, Any]:
    """
    Parse a resume PDF into cleaned text and structured sections.

    Args:
        file_bytes: Raw PDF bytes from upload.

    Returns:
        Dict with ``raw_text`` and ``sections`` keys.
    """
    raw_text = extract_text_from_pdf(file_bytes)
    return parse_resume_text(raw_text)


def parse_resume_text(text: str) -> Dict[str, Any]:
    """
    Parse pasted resume text into cleaned text and structured sections.

    Args:
        text: Raw resume text from a textarea or other text source.

    Returns:
        Dict with ``raw_text`` and ``sections`` keys.
    """
    cleaned_text = clean_text(text)
    if not cleaned_text:
        raise ResumeParsingError("Resume text cannot be empty.")

    sections = segment_sections(cleaned_text)
    logger.info("Resume parsed successfully.")
    return {"raw_text": cleaned_text, "sections": sections}
