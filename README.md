# Simplified Recursive Language Model (RLM)

This project **(src/rlm.py)** is a simplified, easy-to-understand implementation of **[Recursive Language Models (arXiv:2512.24601)](https://arxiv.org/html/2512.24601v1)**.

> **Note:** This repository is an independent, simplified implementation designed for educational purposes and ease of understanding.

> The **official implementation** by the paper authors (Alex L. Zhang, Tim Kraska, Omar Khattab) can be found here: **[https://github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm)**.

RLM allows a Large Language Model (LLM) to programmatically examine, decompose, and recursively call itself over specific snippets of a large document via a Python REPL (Read-Eval-Print Loop), effectively breaking down complex queries into manageable sub-tasks.

## Technical Architecture

Based on the methodology from arXiv:2512.24601, this implementation follows a recursive inference strategy that separates the "Root" planner from "Sub-Model" workers:

1. **Environment Setup**: Load large documents (like patient records) into a Python REPL as a `context` variable, keeping them separate from the main model's memory.
2. **Task Manager**: A main LLM analyzes your question and writes Python code to examine the document in parts, using techniques like text slicing or searching.
3. **Sub-Tasks**: For complex questions, the main LLM breaks the task into smaller pieces, using helper functions to analyze specific document sections.
4. **Combine Results**: The main LLM gathers all partial results and combines them into one final, comprehensive answer.

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

## Citation

If you use this project in your research, please cite the original paper:

```bibtex
@article{zhang2025recursive,
  title={Recursive Language Models},
  author={Zhang, Alex L. and Kraska, Tim and Khattab, Omar},
  journal={arXiv preprint arXiv:2512.24601},
  year={2025},
  url={[https://arxiv.org/abs/2512.24601](https://arxiv.org/abs/2512.24601)}
}
```
