"""Local Ollama LLM factory, tools, and the tool-using GAIA agent (debate agents, aggregator).

All model calls go to the local Ollama server (host from the OLLAMA_HOST env var, default http://localhost:11434).
Only web search (Tavily) and the one-time GAIA download touch the internet.
"""

import glob
import os
from functools import cache

import ollama
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from markitdown import MarkItDown

from gaia_data import GAIA_DIR

# Ollama tags. Trust allocator: one of TRUST_MODELS. Agents: any of AGENT_MODELS. The two choices are independent.
TRUST_MODELS = ("muse-glimmer:30b", "gemma4:31b", "qwen3.8:27b")
AGENT_MODELS = TRUST_MODELS + ("gemma4:12b", "qwen3.5:9b")

MAX_CHARS = 15_000  # cap on every tool output / attachment text so multi-step runs fit in the context window
# Run-wide settings, overwritten from the CLI by the runner.
# num_ctx matters: Ollama's default window is tiny and silently drops the oldest messages (question, system prompt).
CFG = {"num_ctx": 32768, "think": True, "temperature": 0.0, "seed": 42, "timeout": 900.0}


@cache
def capabilities(model: str) -> tuple:
    """Capabilities Ollama reports for a model, e.g. ('completion', 'tools', 'thinking'); () if unknown."""
    try:
        return tuple(ollama.Client().show(model).capabilities or ())
    except Exception:
        return ()


def make_llm(model: str, role: str) -> ChatOllama:
    """`role` (agent_call | trust_audit | corruption) goes into metadata so the trace recorder can label the call."""
    caps = capabilities(model)
    think = CFG["think"] and ("thinking" in caps or not caps)  # models without thinking support reject think=True
    return ChatOllama(model=model, reasoning=think, temperature=CFG["temperature"], seed=CFG["seed"],
                      num_ctx=CFG["num_ctx"], client_kwargs={"timeout": CFG["timeout"]}, metadata={"role": role})


# ---------------------------------------------------------------------------------------------- tools
def file_text(path_or_url: str) -> str:
    """Local file (pdf, docx, xlsx, pptx, csv, json, zip, html, ...) or web page as markdown; errors come back as text."""
    try:
        if "://" not in path_or_url and not os.path.exists(path_or_url):  # the model garbled the path: find by file name
            hits = glob.glob(str(GAIA_DIR / "**" / os.path.basename(path_or_url)), recursive=True)
            path_or_url = hits[0] if hits else path_or_url
        return MarkItDown().convert(path_or_url).text_content[:MAX_CHARS]
    except Exception as e:
        return f"ERROR reading {path_or_url}: {type(e).__name__}: {e}"


@tool
def read_file(path_or_url: str) -> str:
    """Read a local file (pdf, docx, xlsx, pptx, csv, json, zip, html, mp3...) or a web page URL as markdown."""
    return file_text(path_or_url)


@tool
def web_search(query: str) -> str:
    """Search the web. Returns titles, URLs and snippets; open a promising page afterwards with read_file(url)."""
    try:
        res = TavilySearch(max_results=5).invoke({"query": query})
        if not isinstance(res, dict) or "results" not in res:
            return str(res)[:MAX_CHARS]
        return "\n\n".join(f"{r['title']} ({r['url']})\n{r['content']}" for r in res["results"])[:MAX_CHARS]
    except Exception as e:  # a failed search must not crash the debate: the agent sees the error and can adapt
        return f"ERROR: web search failed: {type(e).__name__}: {e}"


SYSTEM = """You are a careful research assistant solving a GAIA benchmark question, possibly as one member of a team that debates the answer.
Work in several steps: plan, call tools, read their results, and keep going until you are sure. Tools:
- web_search(query): find facts; then open a page with read_file(url).
- read_file(path_or_url): read an attachment (pdf, docx, xlsx, pptx, csv, json, zip, html...) or a web page as markdown.
- Python_REPL: run Python for any calculation or data processing (state persists between calls; always print() what you want to see). Never do arithmetic in your head.
If a tool returns an error, fix the call or try another approach.
Finish with a concise, auditable explanation (key evidence and calculations) and a last line `FINAL ANSWER: ...`, where the answer is a number (no units or commas unless asked), a few words, or a comma-separated list."""


def make_agent(model: str):
    """A LangGraph ReAct agent: the model may call tools repeatedly (multi-step) before it answers."""
    return create_agent(make_llm(model, "agent_call"), [web_search, read_file, PythonREPLTool()], system_prompt=SYSTEM)
