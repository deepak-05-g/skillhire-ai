"""Focused alias-resolution tests for the skill extractor."""

import pytest

from app.services.skill_extractor import extractor


@pytest.mark.parametrize(
    "text,expected_skill",
    [
        ("Built UI components with reactjs and hooks.", "React"),
        ("Frontend developer experienced in js and ES6.", "JavaScript"),
        ("Stored documents in mongo for the backend service.", "MongoDB"),
    ],
    ids=["reactjs_to_react", "js_to_javascript", "mongo_to_mongodb"],
)
def test_skill_alias_maps_to_canonical(text: str, expected_skill: str) -> None:
    """Aliases reactjs, js, and mongo should resolve to canonical skill names."""
    result = extractor.extract(text)
    assert expected_skill in result["skills"], (
        f"Expected '{expected_skill}' from text '{text}', got {result['skills']}"
    )
