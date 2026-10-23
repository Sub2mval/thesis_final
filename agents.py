"""Local Ollama LLM factory, tools, and the tool-using GAIA agent (debate agents, aggregator).

All model calls go to the local Ollama server (host from the OLLAMA_HOST env var, default http://localhost:11434).
Only web search (Tavily) and the one-time GAIA download touch the internet.
"""

import functools
import glob
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from functools import cache

import ollama
import requests
from langchain.agents import create_agent
from langchain.agents.middleware import ContextEditingMiddleware
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from langchain_ollama import ChatOllama
from markitdown import MarkItDown

from gaia_data import GAIA_DIR

# Ollama tags. Trust allocator: one of TRUST_MODELS. Agents: any of AGENT_MODELS. The two choices are independent.
TRUST_MODELS = ("muse-glimmer:30b", "gemma4:31b", "qwen3.8:27b")
AGENT_MODELS = TRUST_MODELS + ("gemma4:12b", "qwen3.5:9b")

os.environ.setdefault("MPLBACKEND", "Agg")  # plt.show() must never wait for a window
MAX_CHARS = 8_000  # one page of tool output (read_file pages through longer documents with `offset`)
# Run-wide settings, overwritten from the CLI by the runner.
# num_ctx matters: Ollama's default window is tiny and silently drops the oldest messages (question, system prompt).
CFG = {"num_ctx": 32768, "think": True, "temperature": 0.0, "seed": 42, "timeout": 900.0, "cloud": True}

TOOL_TIMEOUT = 120.0  # seconds one tool call may take; Tavily, URL fetching and the Python shell have no timeout of their own
TOOL_HOOK = None  # optional callable(event, tool_name, info), set by the runner to log tool starts, ends and abandoned calls

# Ollama Cloud (the default backend): only gemma4:31b is served there, under the tag below. Several keys share the load.
CLOUD_HOST = os.getenv("OLLAMA_CLOUD_HOST", "https://ollama.com")  # (overridable so the tests can point it at a fake)
CLOUD_MODELS = {"gemma4:31b": "gemma4:31b-cloud"}
SKIP_KEY_CODES = {401, 403, 429, 529}  # bad key / rate limited / overloaded -> try the next key


def numbered_keys(prefix: str) -> list[str]:
    """PREFIX, PREFIX_1, PREFIX_2, ... from the environment (.env), quotes stripped, duplicates dropped, order kept."""
    keys = [os.getenv(n, "").strip().strip("\"'") for n in [prefix] + [f"{prefix}_{i}" for i in range(1, 100)]]
    return list(dict.fromkeys(k for k in keys if k))


_ollama_key = 0  # round-robin position over OLLAMA_API_KEY, _1, _2, ...


class CloudChatOllama(ChatOllama):
    """ChatOllama on Ollama Cloud with key rotation: every call starts on the next key; a key that is rate limited or
    rejected (401/403/429/529) is skipped; when every key is limited, wait (5 s, 10 s, ... max 120 s) and go round again."""

    def _generate(self, messages, *a, **kw):
        global _ollama_key
        keys = numbered_keys("OLLAMA_API_KEY")
        for rnd in range(10):
            for _ in keys:
                key, _ollama_key = keys[_ollama_key % len(keys)], _ollama_key + 1
                self._client = ollama.Client(host=CLOUD_HOST, headers={"Authorization": f"Bearer {key}"}, timeout=CFG["timeout"])
                try:
                    return super()._generate(messages, *a, **kw)
                except ollama.ResponseError as e:
                    if e.status_code not in SKIP_KEY_CODES:
                        raise
            time.sleep(min(5 * 2 ** rnd, 120))
        raise RuntimeError(f"All {len(keys)} Ollama keys are limited or invalid (tried 10 rounds)")


@cache
def capabilities(model: str) -> tuple:
    """Capabilities Ollama reports for a model, e.g. ('completion', 'tools', 'thinking'); () if unknown."""
    try:
        return tuple(ollama.Client().show(model).capabilities or ())
    except Exception:
        return ()


def make_llm(model: str, role: str) -> ChatOllama:
    """`role` (agent_call | trust_audit | corruption) goes into metadata so the trace recorder can label the call."""
    common = dict(temperature=CFG["temperature"], seed=CFG["seed"], num_ctx=CFG["num_ctx"], metadata={"role": role})
    if CFG["cloud"]:
        return CloudChatOllama(model=CLOUD_MODELS[model], base_url=CLOUD_HOST, reasoning=CFG["think"], **common)
    caps = capabilities(model)
    think = CFG["think"] and ("thinking" in caps or not caps)  # models without thinking support reject think=True
    return ChatOllama(model=model, reasoning=think, client_kwargs={"timeout": CFG["timeout"]}, **common)


# ---------------------------------------------------------------------------------------------- tools
def guarded(fn):
    """Tool wrapper: runs the call in a thread and gives up after TOOL_TIMEOUT s, returning an ERROR text the agent can react
    to (before, one hung search / download / `input()` / endless loop froze the whole run silently). An abandoned call is
    left behind; if it was the Python shell, its stdout capture is undone so the program's own output keeps working."""
    @functools.wraps(fn)
    def run(*a, **kw):
        res, stdout, t0 = {}, sys.stdout, time.time()
        arg = str((list(kw.values()) + list(a) or [""])[0]).replace("\n", " ")[:80]
        TOOL_HOOK and TOOL_HOOK("start", fn.__name__, arg)

        def work():
            try:
                res["out"] = fn(*a, **kw)
            except BaseException as e:  # noqa: BLE001 - reported to the caller below
                res["err"] = e
        th = threading.Thread(target=work, daemon=True)
        th.start()
        th.join(TOOL_TIMEOUT)
        if th.is_alive():
            sys.stdout = stdout
            TOOL_HOOK and TOOL_HOOK("timeout", fn.__name__, f"{time.time() - t0:.0f}s")
            return (f"ERROR: {fn.__name__} did not finish within {TOOL_TIMEOUT:g} s and was abandoned (endless loop, waiting for "
                    "input, or a stalled network request). Try a simpler or different approach.")
        if "err" in res:
            raise res["err"]
        TOOL_HOOK and TOOL_HOOK("end", fn.__name__, f"{time.time() - t0:.1f}s, {len(str(res['out']))} chars")
        return res["out"]
    return run


def _no_input(*a, **kw):
    raise EOFError("input() is not available: nobody can type; put the values in the code")


@cache
def _convert(path_or_url: str) -> str:
    """Full text of a local file (pdf, docx, xlsx, pptx, csv, json, zip, html, ...) or web page, converted once per run."""
    if "://" not in path_or_url and not os.path.exists(path_or_url):  # the model garbled the path: find by file name
        hits = glob.glob(str(GAIA_DIR / "**" / os.path.basename(path_or_url)), recursive=True)
        path_or_url = hits[0] if hits else path_or_url
    return MarkItDown().convert(path_or_url).text_content


def file_text(path_or_url: str, offset: int = 0) -> str:
    """One page (MAX_CHARS) of the document as markdown. If there is more, the page says so and how to get it,
    so truncation is never silent. Errors come back as text (the agent can react to them)."""
    try:
        text = _convert(path_or_url)
    except Exception as e:
        return f"ERROR reading {path_or_url}: {type(e).__name__}: {e}"
    page = text[offset:offset + MAX_CHARS]
    end = offset + len(page)
    more = f"\n[showing characters {offset}-{end} of {len(text)}; call read_file with offset={end} for the next page]" if end < len(text) else ""
    return page + more


@tool
@guarded
def read_file(path_or_url: str, offset: int = 0) -> str:
    """Read a local file (pdf, docx, xlsx, pptx, csv, json, zip, html, mp3...) or a web page URL as markdown, one page at a time.
    Long documents continue at `offset` (the page tells you which)."""
    return file_text(path_or_url, offset)


TAVILY_SKIP_CODES = {401, 403, 429, 432, 433}  # bad key / rate limited / plan or credit limit exceeded -> next key
_key = 0  # index of the Tavily key in use; sticks after a rotation so exhausted keys are not retried first


def tavily_keys() -> list[str]:
    """TAVILY_API_KEY, TAVILY_API_KEY_1, _2, ... (the free tier is 1000 credits per key and a full run needs several
    thousand searches, so a quota error moves on to the next key)."""
    return numbered_keys("TAVILY_API_KEY")


@tool
@guarded
def web_search(query: str) -> str:
    """Search the web. Returns titles, URLs and snippets; open a promising page afterwards with read_file(url)."""
    global _key
    keys = tavily_keys()
    for _ in range(max(len(keys), 1)):
        key = keys[_key % len(keys)] if keys else ""
        try:  # plain REST call with a timeout (the langchain wrapper had none)
            r = requests.post("https://api.tavily.com/search", json={"query": query, "max_results": 5},
                              headers={"Authorization": f"Bearer {key}"}, timeout=30)
        except requests.RequestException as e:  # a failed search must not crash the debate: the agent sees the error
            return f"ERROR: web search failed: {type(e).__name__}: {e}"
        if r.status_code in TAVILY_SKIP_CODES and len(keys) > 1:  # key exhausted / rejected -> next key
            _key += 1
            continue
        if r.status_code != 200:
            return f"ERROR: web search failed: HTTP {r.status_code} {r.text[:200]}"
        hits = r.json().get("results") or []
        return ("\n\n".join(f"{h['title']} ({h['url']})\n{h['content']}" for h in hits) or "(no results)")[:MAX_CHARS]
    return f"ERROR: web search failed: all {len(keys)} Tavily keys are exhausted or rejected"


SYSTEM = """You are a careful research assistant solving a GAIA benchmark question, possibly as one member of a team that debates the answer.
Work in several steps: plan, call tools, read their results, and keep going until you are sure. Tools:
- web_search(query): find facts; then open a page with read_file(url).
- read_file(path_or_url, offset=0): read an attachment (pdf, docx, xlsx, pptx, csv, json, zip, html...) or a web page as markdown, one page at a time; long documents continue at the offset the page names.
- Python_REPL: run Python for any calculation or data processing (state persists between calls; always print() what you want to see). Never do arithmetic in your head.
You have roughly 15 tool calls per turn: stop researching and answer before they run out. Older tool outputs are shortened to a preview to save space, so write down the facts you need as you find them. If a tool returns an error, fix the call or try another approach.
Finish with a concise, auditable explanation (key evidence and calculations) and a last line `FINAL ANSWER: ...`, where the answer is a number (no units or commas unless asked), a few words, or a comma-separated list."""


def python_tool():
    """Stateful Python shell (state persists between calls of one agent session) with capped output and a print() hint."""
    repl = PythonREPLTool()
    repl.python_repl.globals["input"] = _no_input

    def Python_REPL(query: str) -> str:
        """Run Python code. State persists between calls. Always print() what you want to see."""
        return (repl.invoke(query) or "(no output - did you forget print()?)")[:MAX_CHARS]
    return tool(guarded(Python_REPL))


@dataclass
class MaskOldObservations:
    """Context edit (applied to a copy: the stored history and the trace stay complete). Newest tool outputs are kept in
    full up to `budget` characters; older ones shrink to a short preview. The agent's own reasoning, actions and answers
    are never touched, so it still knows what it did and learned. Deterministic (no token counting), so once an output is
    masked it stays masked and the prompt prefix stays stable for Ollama's KV cache."""
    budget: int
    preview: int = 300

    def apply(self, messages, *, count_tokens=None) -> None:
        used = 0
        for i in range(len(messages) - 1, -1, -1):
            m = messages[i]
            if not isinstance(m, ToolMessage):
                continue
            text = str(m.content)
            used += len(text)
            if used > self.budget and len(text) > self.preview:
                hidden = f"\n[... {len(text) - self.preview} more characters hidden to save context; call the tool again if you need them]"
                messages[i] = m.model_copy(update={"content": text[:self.preview] + hidden})


def make_agent(model: str):
    """A LangGraph ReAct agent: the model may call tools repeatedly (multi-step) before it answers.
    Tool outputs are what fills the context window (Ollama then truncates silently or cuts generation mid-answer), so the
    newest ones get ~1.2 characters per context token (about a third of the window) and older ones are masked."""
    return create_agent(make_llm(model, "agent_call"), [web_search, read_file, python_tool()], system_prompt=SYSTEM,
                        middleware=[ContextEditingMiddleware(edits=[MaskOldObservations(budget=int(CFG["num_ctx"] * 1.2))])])
