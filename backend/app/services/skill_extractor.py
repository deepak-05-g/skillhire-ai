"""Skill extraction from resume/job text using a taxonomy and alias map."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Pattern, Set, Tuple

logger = logging.getLogger(__name__)


class SkillExtractor:
    """Extract canonical technical skills from free text."""

    def __init__(self) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.skills_path = os.path.abspath(
            os.path.join(current_dir, "..", "..", "..", "data", "skills.json")
        )

        self.skills_db = self._load_skills_db()
        self.canonical_skills: List[str] = []
        for category_skills in self.skills_db.values():
            self.canonical_skills.extend(category_skills)

        self.aliases: Dict[str, str] = {
            "js": "JavaScript",
            "node": "Node.js",
            "reactjs": "React",
            "mongo": "MongoDB",
            "ml": "Machine Learning",
            "nlp": "NLP",
        }

        self.patterns: Dict[str, Pattern[str]] = {
            skill: self._compile_pattern(skill) for skill in self.canonical_skills
        }
        self.alias_patterns: Dict[str, Tuple[Pattern[str], str]] = {
            alias: (self._compile_pattern(alias), canonical)
            for alias, canonical in self.aliases.items()
        }

    def _load_skills_db(self) -> Dict[str, List[str]]:
        """Load skill taxonomy from JSON, falling back to built-in defaults."""
        try:
            if os.path.exists(self.skills_path):
                with open(self.skills_path, encoding="utf-8") as handle:
                    data = json.load(handle)
                    logger.info("Loaded skills taxonomy from %s", self.skills_path)
                    return data
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Could not load skills taxonomy at %s (%s). Using fallback list.",
                self.skills_path,
                exc,
            )

        return {
            "Programming": ["Python", "Java", "C", "C++", "JavaScript", "TypeScript"],
            "Frontend": ["HTML", "CSS", "React", "Next.js", "Tailwind CSS"],
            "Backend": ["Node.js", "Express.js", "FastAPI", "Django", "Flask"],
            "Database": ["SQL", "MySQL", "PostgreSQL", "MongoDB"],
            "ML/Data": [
                "Machine Learning", "Deep Learning", "NLP", "Pandas", "NumPy",
                "Scikit-learn", "TensorFlow", "PyTorch", "Matplotlib", "Seaborn",
            ],
            "Cloud/DevOps": [
                "Git", "GitHub", "Docker", "AWS", "Azure", "Google Cloud",
                "Linux", "REST API",
            ],
            "CS": ["DSA", "OOP", "DBMS", "Operating Systems", "Computer Networks"],
        }

    def _compile_pattern(self, term: str) -> Pattern[str]:
        """Build a boundary-safe regex for a skill term or alias."""
        escaped = re.escape(term)
        prefix = r"\b" if re.match(r"^\w", term) else r""
        suffix = r"\b" if re.match(r".*\w$", term) else r""
        return re.compile(f"{prefix}{escaped}{suffix}", re.IGNORECASE)

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract matched skills from text and resolve aliases to canonical names.

        Args:
            text: Resume or job posting text.

        Returns:
            Dict with sorted ``skills`` list and ``skill_count``.
        """
        if not text:
            return {"skills": [], "skill_count": 0}

        matched_skills: Set[str] = set()

        for skill, pattern in self.patterns.items():
            if pattern.search(text):
                matched_skills.add(skill)

        for _alias, (pattern, canonical) in self.alias_patterns.items():
            if pattern.search(text):
                matched_skills.add(canonical)

        sorted_skills = sorted(matched_skills)
        logger.debug("Extracted %d skills from text.", len(sorted_skills))
        return {"skills": sorted_skills, "skill_count": len(sorted_skills)}


extractor = SkillExtractor()
