"""Shared helpers and assertions for SkillHire AI tests."""

from typing import Any, Dict, Iterable

# Common normalized job schema shared by Greenhouse, Lever, and Ashby connectors.
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


def assert_common_job_format(job: Dict[str, Any]) -> None:
    """
    Verify a job dict matches the normalized schema used across job sources.

    Raises:
        AssertionError: If required fields are missing or have wrong types.
    """
    missing = [field for field in COMMON_JOB_FIELDS if field not in job]
    assert not missing, f"Normalized job missing fields: {', '.join(missing)}"

    assert isinstance(job["source"], str) and job["source"], "source must be a non-empty string"
    assert isinstance(job["company"], str) and job["company"], "company must be a non-empty string"
    assert isinstance(job["title"], str) and job["title"], "title must be a non-empty string"
    assert isinstance(job["location"], str), "location must be a string"
    assert isinstance(job["description"], str), "description must be a string"
    assert isinstance(job["requirements"], str), "requirements must be a string"
    assert isinstance(job["apply_url"], str), "apply_url must be a string"
    assert isinstance(job["job_type"], str), "job_type must be a string"


def assert_all_common_job_format(jobs: Iterable[Dict[str, Any]]) -> None:
    """Assert every job in an iterable matches the common normalized format."""
    jobs_list = list(jobs)
    assert jobs_list, "Expected at least one normalized job"
    for index, job in enumerate(jobs_list):
        try:
            assert_common_job_format(job)
        except AssertionError as exc:
            raise AssertionError(f"Job at index {index} failed validation: {exc}") from exc
