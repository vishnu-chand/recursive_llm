# RLM (Recursive Language Model) Agent

RLM is a powerful language model agent designed for processing and analyzing large text documents with a focus on medical records and patient history. It uses recursive techniques to efficiently process text in chunks while maintaining context awareness.

## Features

- Processes large documents by breaking them into manageable chunks
- Maintains context across document sections
- Supports semantic search and information extraction
- Customizable chunking and processing parameters
- Built-in caching for improved performance
- Comprehensive test suite with unit and integration tests
- Pre-configured development environment with linting and formatting

## Project Structure

```
.
├── src/                                    # Source code directory
│   ├── rlm.py                              # Main RLM implementation
│   ├── app.py                              # Main application entry point
│   └── temp.py                             # Temporary/utility scripts
├── tests/                                  # Test files
│   ├── __init__.py                         # Test package initialization
│   ├── test_app_utils.py                   # Tests for application utilities
│   ├── test_rlm_environment.py             # Tests for RLM environment
│   ├── test_recursive_language_model.py    # Tests for RLM core
│   └── conftest.py                         # Test fixtures and configurations
├── data/                                   # Data files
│   └── synthetic_patient_data.md           # Sample patient data
├── requirements.txt                        # Python dependencies
└── Dockerfile                              # Container configuration
```

## Commands

### Clone and build

```bash
# git clone <repository-url>
cd recursive_llm/src
# lsof -t -i:7860 | xargs -r kill -9
# podman machine stop && podman machine start
podman build -t rlm-app .
podman run -it --rm -p 7860:7860 -e GOOGLE_API_KEY rlm-app
```

### Run with custom settings

```bash
podman run -it --rm -p 7860:7860 -e GOOGLE_API_KEY rlm-app
```

## Usage

### Basic Usage

```python
from rlm import RecursiveLanguageModel

# Initialize the RLM with your preferred model
rlm = RecursiveLanguageModel("gpt-4")  # or your preferred model name

# Load your document
with open('path/to/document.txt', 'r') as f:
    context = f.read()

# Ask a question
question = "What are the key medical issues in this patient's history?"
result = rlm.run(
    context=context,
    messages=[{"role": "user", "content": question}],
    max_chunk_size=10000,
    max_output_chars=10000,
    max_steps=15
)

print(result)
```

## Testing

The project includes a comprehensive test suite to ensure code quality and reliability.

### Running Tests

1. Install test dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the test suite:
   ```bash
   pytest -v
   pytest --cov=src --cov-report=term-missing
   pytest tests/test_rlm_environment.py -v
   pytest -k "test_llm_query" -v
   ```

### Test Structure

- `test_rlm_environment.py`: Tests for the RLM environment and core functionality
- `test_recursive_language_model.py`: Tests for the main RLM class
- `test_app_utils.py`: Tests for application utilities
- `conftest.py`: Shared test fixtures and configurations
