"""Pytest configuration: prevent real LLM calls during tests."""
from cogniteam.config.settings import settings

settings.use_ollama = False
