# Simplified Recursive Language Model (RLM)

This project **(src/rlm.py)** is a simplified, easy-to-understand implementation of **[Recursive Language Models (arXiv:2512.24601)](https://arxiv.org/html/2512.24601v1)**.

It demonstrates the core concept of utilizing inference-time scaling to handle potentially infinite context lengths by treating text as an external environment rather than filling the context window.

> **Note:** This repository is an independent, simplified implementation designed for educational purposes and ease of understanding.

> The **official implementation** by the paper authors (Alex L. Zhang, Tim Kraska, Omar Khattab) can be found here: **[https://github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm)**.

RLM allows a Large Language Model (LLM) to programmatically examine, decompose, and recursively call itself over specific snippets of a large document via a Python REPL (Read-Eval-Print Loop), effectively breaking down complex queries into manageable sub-tasks.

## Features

- **Infinite Context Processing**: Handles inputs far beyond standard context windows (10M+ tokens) by treating the prompt as an external variable within a REPL environment.
- **Inference-Time Scaling**: Turns long-context processing into an algorithmic search problem, scaling compute at inference time rather than just expanding the context window.
- **Recursive Decomposition**: The model generates code to slice the document and recursively invokes sub-instances of itself on relevant chunks, enabling a "divide-and-conquer" approach for complex reasoning.
- **REPL-Based Execution**: Utilizes a persistent Python sandbox to execute reasoning steps, manage variables, perform regex searches, and stitch together results.
- **Context Rot Mitigation**: Avoids performance degradation common in long-context models by selectively retrieving only the necessary information for each sub-task.
- **Cost Efficiency**: Often cheaper than processing full contexts or standard RAG approaches because the model selectively "reads" data rather than ingesting it all at once.

## Technical Architecture

Based on the methodology from arXiv:2512.24601, this implementation follows a recursive inference strategy that separates the "Root" planner from "Sub-Model" workers:

1.  **Environment Setup (The Control Plane)**: The large context (e.g., patient history) is loaded into a sandboxed Python REPL environment as a string or list variable (e.g., `context`). It is *never* passed directly to the Root LLM's context window.
2.  **Root Planner**: A "Root" LLM receives the user query and a system prompt explaining available tools. It acts as an orchestrator, writing Python code to inspect the context variable using slicing (e.g., `context[:1000]`), keyword searches, or regex matching.
3.  **Recursive Sub-Calls**: To answer complex questions, the Root LLM defines sub-tasks by slicing the context and passing these chunks to a `llm_query` (or `llm_batch` for parallel processing) function. This spawns a **Sub-Model**—which can be a smaller model instance or the same model—to process just that specific chunk.
4.  **Aggregation**: The results from these recursive sub-calls are returned to the REPL as variables. The Root LLM then reads these intermediate outputs and aggregates them into a final answer, effectively synthesizing information from across the entire document without ever seeing it all at once.

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
