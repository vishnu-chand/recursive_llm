import pytest
from unittest.mock import MagicMock, patch


# This file is used to define fixtures and other test configurations
# that will be available to all tests in the test suite.


@pytest.fixture
def mock_llm_response():
    """Fixture that provides a mock LLM response."""

    def _create_mock_response(content: str):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = content
        return mock_response

    return _create_mock_response


@pytest.fixture
def sample_context():
    """Fixture that provides a sample context for testing."""
    return """This is a test context with some information.
    It contains multiple lines of text for testing purposes.
    The quick brown fox jumps over the lazy dog.
    """


@pytest.fixture
def sample_messages():
    """Fixture that provides sample chat messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the test about?"},
    ]


# Add any additional fixtures here
