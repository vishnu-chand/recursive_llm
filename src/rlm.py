import io
import json
import litellm
import logging
import warnings
import traceback
import contextlib
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


def get_logger(name):
    class MultiLineFormatter(logging.Formatter):
        def format(self, record):
            record.location = f"{record.filename[:12]}:{record.funcName[:12]}:{record.lineno}"
            output = super().format(record)
            header = output.split(record.message)[0]
            return output.replace('\n', '\n' + header)

    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.propagate = False

    handler = logging.StreamHandler()
    handler.setFormatter(MultiLineFormatter(fmt='[%(levelname).1s %(asctime)s | %(location)-30s] %(message)s', datefmt='%y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)
    return logger


logger = get_logger("RLM")
logger.setLevel(logging.DEBUG)
litellm.cache = litellm.Cache(type="disk")
system_prompt = r"""
You are an RLM (Recursive Language Model) Agent.
You are an inference-time algorithm capable of processing infinite text by treating it as an external environment variable `CONTEXT`.

**ENVIRONMENT:**
- `CONTEXT` (str): The full input text. NEVER print, load, or pass this entire variable to an LLM. Access via slicing `CONTEXT[start:end]`.
- `llm_query(query, snippet) -> str`: Spawns a sub-agent on a specific chunk. Returns extracted answers or an empty string.
- `FINAL(answer)`: Submits the final answer and exits. Every code should call FINAL.
- `print(val)`: Standard output for debugging logic.
- Max Chunk Size: {max_chunk_size}
- Max Output Chars: {max_output_chars} 
- Context Size: Context has {context_size} chars
- Context Sample: 
    {context_sample}...

**CORE PHILOSOPHY:**
1. **Semantic Expansion (Broad Search):** Do not just search for words present in the user query. Decompose the user's intent into a list of "weakly related" keywords, proxies, dates, or technical concepts that *might* appear in the text.
2. **Iterative Narrowing:** 
   - Start with a broad set of keywords.
   - Do smart pre-processing or chunking.
   - Use `llm_query` on chunks/snippets to filter and extract details relevant to query.
   - If results are sparse, expand the keyword list further.
3. **STRICT CONSTRAINTS:**
   - **Chunk Size:** NEVER create chunks larger than `max_chunk_size`. Always check `max_chunk_size` before setting `chunk_size`.
   - **Output Length:** Ensure any output, including intermediate results, stays under `max_output_chars`.
   - **Context Handling:** `FINAL(llm_query(prompt, CONTEXT))` is strictly forbidden. You must slice `CONTEXT` into appropriate chunks.
4. **Efficiency:**
   - Process only relevant chunks of text
   - Use overlapping chunks to prevent missing information at boundaries
   - Implement early termination when possible to save resources

**EXAMPLE 1**
Query: "Identify all episodes of suspected Sepsis in the patient's history."

# RLM Agent Generation:

# 1. EXPAND
# "Sepsis" might not be explicitly written. I need clinical criteria (Broad Search).
proxies = [
    "fever", "temp > 38", "hypotension", "BP < 90",  # Vitals
    "tachycardia", "HR > 100",                       # Vitals
    "white blood cell", "WBC", "leukocytosis",       # Labs
    "blood culture", "lactate",                      # Diagnostics
    "vancomycin", "piperacillin", "bolus"            # Treatments
]

# 2. CHUNK & SEARCH
import re

overlap = 100
chunk_size = {max_chunk_size} # try to keep chunk_size around `Max Chunk Size`
relevant_findings = set()
# Iterate by index to avoid loading full CONTEXT into memory
for i in range(0, len(CONTEXT), chunk_size - overlap):
    # Slice a specific window
    snippet = CONTEXT[i : i + chunk_size]
    
    # Pre-filter: Only process chunks that contain valid terms
    if any(re.search(re.escape(term), snippet, re.IGNORECASE) for term in proxies):
        # 3. EXTRACT via Sub-Agent
        note = llm_query(f"Extract date and description of sepsis events here.", snippet)
        if note:
            # try to keep len(note) smaller than `Max Output Chars`
            relevant_findings.add(note)

# 4. SYNTHESIZE
if relevant_findings:
    FINAL(llm_query("Create a timeline of these sepsis episodes:", "\n".join(relevant_findings)))
else:
    FINAL("No sepsis episodes found in the provided history.")

**EXAMPLE 2**
Query: "short summary of patient's history."

# RLM Agent Generation:

# 1. STRATEGY: Map-Reduce
# Summarization requires global context, so I cannot filter by keywords.
# I must process the text linearly in chunks (Map) and combine results (Reduce).

chunk_size = {max_chunk_size} # try to keep chunk_size around `Max Chunk Size`
chunk_summaries = []

# 2. MAP (Process Chunks)
for i in range(0, len(CONTEXT), chunk_size):
    snippet = CONTEXT[i : i + chunk_size]
    
    # Skip empty or noise chunks
    if len(snippet.strip()) < 50:
        continue
        
    # Compress the chunk into a key point
    mini_sum = llm_query("Extract key medical events and dates from this section. Be brief.", snippet)
    if mini_sum:
        # try to keep len(mini_sum) smaller than `Max Output Chars`
        chunk_summaries.append(mini_sum)

# 3. REDUCE (Synthesize)
combined_notes = "\n".join(chunk_summaries)

if not combined_notes:
    FINAL("The document appears to be empty or contains no medical history.")
else:
    # Final pass to ensure flow and brevity
    FINAL(llm_query("Write a cohesive 1-paragraph summary based on these notes.", combined_notes))
""".strip()
sub_agent_prompt = r"""
You are a precise analytic sub-agent.

CONTEXT SNIPPET:
---
{context_snippet}

TASK:
---
{prompt}

INSTRUCTIONS:
1. Analyze if the context snippet contains task specific relevant info.
2. Process only the relevant information.
3. Return the result in the required JSON structure.
""".strip()


def disp_messages(messages):
    logger.info(f"_______________________________________________________________")
    for m in messages[1:]:
        logger.info(f"******************************************************************")
        role, content = m['role'], m['content']
        try:
            content = json.loads(content)
            logger.info(f"[ASSISTANT THOUGHT]")
            logger.info(content['thought'])
            logger.info(f"[ASSISTANT CODE]")
            logger.info(content['code'])
        except:
            logger.info(f"[{role.upper()}]")
            logger.info(content)


class AgentAction(BaseModel):
    thought: str
    code: str


class SnippetAnalysis(BaseModel):
    is_relevant: bool = Field(..., description="True if the snippet contains information answering the prompt.")
    answer: str = Field(..., description="The synthesized answer based ONLY on the snippet.")


class RLMEnvironment:
    def __init__(self, context, model_name, max_output_chars):
        self.answer = None
        self.context = context
        self.model_name = model_name
        self.max_output_chars = max_output_chars
        self.globals = {"CONTEXT": context, "llm_query": self.llm_query, "FINAL": self.final, "print": print, "len": len, "range": range, "enumerate": enumerate, }

    def llm_query(self, prompt, context_snippet):
        content = sub_agent_prompt.format(prompt=prompt, context_snippet=context_snippet)
        response = litellm.completion(model=self.model_name, messages=[{"role": "user", "content": content}], response_format=SnippetAnalysis, caching=True)
        content = response.choices[0].message.content
        analysis = SnippetAnalysis.model_validate_json(content)
        output = analysis.model_dump()
        output = output['answer'] if output['is_relevant'] else ''
        return output

    def final(self, answer):
        self.answer = answer
        print(f"{answer}")
        return "Task Completed."

    def execute(self, code):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(code, self.globals)
        output = buffer.getvalue()
        if len(output) > self.max_output_chars:
            output = output[:self.max_output_chars] + f"\n... [Output truncated. Length: {len(output)}]"
        return output


class RecursiveLanguageModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def run(self, context, messages, max_chunk_size, max_output_chars, max_steps):
        env = RLMEnvironment(context, self.model_name, max_output_chars)
        prompt = system_prompt.format(context_size=len(context), context_sample=context[:200], max_output_chars=max_output_chars, max_chunk_size=max_chunk_size)
        messages = [{"role": "system", "content": prompt}, *messages]
        for step in range(max_steps):
            if env.answer:
                break
            try:
                response = litellm.completion(model=self.model_name, messages=messages, response_format=AgentAction, caching=True)
                action_data = AgentAction.model_validate_json(response.choices[0].message.content)
                messages.append(response.choices[0].message)
                output = env.execute(action_data.code)
                messages.append({"role": "user", "content": f"Execution Output: {output}"})
            except:
                messages.append({"role": "user", "content": f"Error: {traceback.format_exc()}"})
        if not env.answer:
            env.answer = "Max steps reached without final answer."
        disp_messages(messages)
        return env.answer


def run_rlm():
    with open('../data/synthetic_patient_data.md', 'r') as book:
        patient_history = book.read()

    rlm = RecursiveLanguageModel("gemini/gemini-2.5-flash")
    # question = "What are the critical issues in this patient's history?"
    # question = "What are the issues related to Nerves?"
    question = "What are the issues related Hair loss?"
    # question = "What is the issues rarest of rare event?"
    result = rlm.run(patient_history, [{"role": "user", "content": question}], 10000, 10000, 15)
    logger.info(f"Final answer: {result}")


if __name__ == "__main__":
    run_rlm()
