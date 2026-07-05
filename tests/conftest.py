"""
Pytest configuration — shared path setup and isolated SQLite database for tests.
"""

import os
import sys

import pytest

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_backend_dir = os.path.join(_root, "backend")
_test_db_path = os.path.join(_backend_dir, "test_skillhire_ai.db")

# Ensure backend package imports work from any test file.
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

os.environ["DATABASE_URL"] = f"sqlite:///{_test_db_path}"


@pytest.fixture(autouse=True)
def reset_database():
    """Reset tables before each test for isolation."""
    from app.database import Base, engine

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield

    if os.path.exists(_test_db_path):
        try:
            os.remove(_test_db_path)
        except OSError:
            pass
