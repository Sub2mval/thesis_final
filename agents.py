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

import httpx
import ollama
import requests
from langchain.agents import create_agent
from langchain.agents.middleware import ContextEditingMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
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
CFG = {"num_ctx": 32768, "think": True, "temperature": 0.0, "seed": 42, "timeout": 900.0, "cloud": True,
       "max_tokens": 12288, "max_key_wait": 7200.0,
       "max_steps": 10, "token_budget": 250_000}
USAGE = {"tokens": 0, "lean": 0}  # tokens spent in the current phase (kept by the runner's recorder); agent turns taken in budget mode

TOOL_TIMEOUT = 120.0  # seconds one tool call may take; Tavily, URL fetching and the Python shell have no timeout of their own
NOTE_HOOK = None  # optional callable(message), set by the runner to log key rotations
TOOL_HOOK = None  # optional callable(event, tool_name, info), set by the runner to log tool starts, ends and abandoned calls

# Ollama Cloud (the default backend): only gemma4:31b is served there, under the tag below. Several keys share the load.
CLOUD_HOST = os.getenv("OLLAMA_CLOUD_HOST", "https://ollama.com")  # (overridable so the tests can point it at a fake)
CLOUD_MODELS = {"gemma4:31b": "gemma4:31b-cloud"}
SKIP_KEY_CODES = {401, 403, 429, 529}  # bad key / rate limited / overloaded -> try the next key


def numbered_keys(prefix: str) -> list[str]:
    """PREFIX, PREFIX_1, PREFIX_2, ... from the environment (.env), quotes stripped, duplicates dropped, order kept."""
    keys = [os.getenv(n, "").strip().strip("\"'") for n in [prefix] + [f"{prefix}_{i}" for i in range(1, 100)]]
    return list(dict.fromkeys(k for k in keys if k))


class KeysExhausted(RuntimeError):
    """Every Ollama key is cooling down for longer than --max-key-wait: stop cleanly (everything done is saved; resume later)."""


# ---- generation guards (all backends) ----------------------------------------------------------------------------------
def shrink(messages, cap: int = 20_000):
    """Last-resort safety net for 'prompt too long': cut any single message above `cap` characters to its head and tail."""
    def cut(text: str) -> str:
        return text if len(text) <= cap else f"{text[:cap // 2]}\n[... {len(text) - cap} characters cut: message far too long ...]\n{text[-cap // 2:]}"

    def fix(m):
        c = m.content if hasattr(m, "content") else m.get("content")
        c = cut(c) if isinstance(c, str) else [{**b, "text": cut(b["text"])} if isinstance(b, dict) and b.get("type") == "text" else b for b in c or []]
        return m.model_copy(update={"content": c}) if hasattr(m, "model_copy") else {**m, "content": c}
    return [fix(m) for m in messages]


class Runaway(Exception):
    """A streamed reply that ran away (endless repetition or far too long); the stream was cut off client-side."""


def _watch(stream):
    """Pass a streamed reply through, but abort it (closing the connection, so the server stops generating) as soon as it
    is clearly a runaway: more than 3 characters per allowed output token, or one 200-character block repeating 6+ times in
    the last 12000 characters. Needed because Ollama Cloud does not seem to honour num_predict: runaway thinking loops
    ran to the server's own 65536-token limit (about 4 minutes and 65k tokens each)."""
    total, since, tail = 0, 0, ""
    try:
        for part in stream:
            seg = (getattr(part.message, "thinking", None) or "") + (part.message.content or "")
            total, since, tail = total + len(seg), since + len(seg), (tail + seg)[-12000:]
            if total > 3 * CFG["max_tokens"]:
                raise Runaway(f"more than {3 * CFG['max_tokens']} characters of output")
            if since >= 1000:
                since = 0
                if len(tail) >= 2400 and tail.count(tail[-200:]) >= 6:
                    raise Runaway(f"endless repetition after {total} characters")
            yield part
    finally:
        getattr(stream, "close", lambda: None)()


class WatchedClient(ollama.Client):
    """ollama.Client whose streamed chat replies go through _watch."""

    def chat(self, *a, **kw):
        stream = super().chat(*a, **kw)
        return _watch(stream) if kw.get("stream") else stream


def done_reason(res) -> str | None:
    """Why generation stopped ('stop' | 'length' | ...). Inside _generate it is in generation_info, not yet in response_metadata."""
    g = res.generations[0]
    return (g.generation_info or {}).get("done_reason") or (getattr(g.message, "response_metadata", None) or {}).get("done_reason")


class GuardedChatOllama(ChatOllama):
    """ChatOllama that never lets a runaway generation or an oversized prompt wreck a run:
    * a reply that runs away (Runaway) or is cut off at the output cap (done_reason=length) is NOT used; it is regenerated
      with different sampling, then without thinking (an identical retry would loop again at temperature 0 with a fixed seed).
      If all three attempts run away the reply is empty, and the caller's own fallbacks (forced answer, unusable audit) take over.
    * a 400 'prompt is too long' is retried once with oversized messages cut down (shrink).
    The key rotation of the cloud subclass sits INSIDE each attempt (_once), so a failed key never restarts the whole sequence."""

    def _once(self, messages, *a, **kw):
        return super()._generate(messages, *a, **kw)

    def _generate(self, messages, *a, **kw):
        def call(obj, msgs):
            try:
                return obj._once(msgs, *a, **kw)
            except ollama.ResponseError as e:
                if e.status_code != 400 or "too long" not in str(e).lower():
                    raise
                NOTE_HOOK and NOTE_HOOK(f"prompt too long ({e}); retrying with oversized messages cut down")
                return obj._once(shrink(msgs), *a, **kw)

        res = None
        for n, change in enumerate(({}, {"temperature": 0.6}, {"temperature": 0.3, "reasoning": False})):
            try:
                res = call(self if n == 0 else self.model_copy(update={**change, "seed": CFG["seed"] + n}), messages)
                if done_reason(res) != "length":
                    return res
                why = "reply hit the output cap"
            except Runaway as e:
                res, why = None, f"runaway generation ({e})"
            NOTE_HOOK and NOTE_HOOK(f"{self.model}: {why}; " + (f"regenerating ({'different sampling' if n == 0 else 'thinking off'}) instead of using it" if n < 2 else "giving up, empty reply"))
        return res or ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])


# ---- Ollama Cloud keys: round robin with per-key cooldowns ------------------------------------------------------------
_ollama_key = 0  # round-robin position over OLLAMA_API_KEY, _1, _2, ...
_cool: dict[str, tuple[float, int]] = {}  # key -> (cooling down until, consecutive failures)
_lock = threading.Lock()


def _pick_key(keys: list[str]) -> tuple[int | None, float]:
    """Next key (round robin) that is not cooling down -> (index, 0); if all are: (None, seconds until the first is back)."""
    global _ollama_key
    with _lock:
        now = time.time()
        for _ in keys:
            i, _ollama_key = _ollama_key % len(keys), _ollama_key + 1
            if _cool.get(keys[i], (0, 0))[0] <= now:
                return i, 0.0
        return None, min(_cool[k][0] for k in keys) - now


def _cool_down(key: str, base: float) -> float:
    """Park a failed key for base * 2^(consecutive failures - 1) seconds (max 2 h); a success resets it. -> seconds."""
    with _lock:
        strikes = _cool.get(key, (0, 0))[1] + 1
        secs = min(base * 2 ** (strikes - 1), max(base, 7200))
        _cool[key] = (time.time() + secs, strikes)
        return secs


class CloudChatOllama(GuardedChatOllama):
    """Guarded ChatOllama on Ollama Cloud, spread over several keys without 429 death spirals: every call goes to the next key
    that is not cooling down; a failing key is parked with a growing cooldown instead of being hammered again by every later
    call (monthly usage limit 6 h, plan usage limit 15 min, other 429/529 20 s, bad key 24 h, stalled connection 30 s; each
    doubling per repeat, max 2 h except the monthly one). When all keys are parked the call sleeps until the first is back; if
    that would exceed CFG max_key_wait it raises KeysExhausted (the run stops cleanly and can be resumed)."""

    def _once(self, messages, *a, **kw):
        keys, waited = numbered_keys("OLLAMA_API_KEY"), 0.0
        while True:
            i, wait = _pick_key(keys)
            if i is None:
                if waited + wait > CFG["max_key_wait"]:
                    raise KeysExhausted(f"all {len(keys)} Ollama keys are rate limited / rejected for another {wait / 60:.0f} min "
                                        f"(--max-key-wait exceeded); rerun the same command later to resume")
                NOTE_HOOK and NOTE_HOOK(f"all {len(keys)} Ollama keys are cooling down; waiting {wait:.0f}s for the first one")
                time.sleep(wait + 1)
                waited += wait + 1
                continue
            self._client = WatchedClient(host=CLOUD_HOST, headers={"Authorization": f"Bearer {keys[i]}"}, timeout=CFG["timeout"])
            try:
                res = super()._once(messages, *a, **kw)
                _cool.pop(keys[i], None)
                return res
            except ollama.ResponseError as e:
                if e.status_code not in SKIP_KEY_CODES:
                    raise
                low = str(e).lower()
                base = 86400 if e.status_code in (401, 403) else 21600 if "monthly" in low else 900 if "usage limit" in low else 20
                why = f"HTTP {e.status_code}"
            except (httpx.TransportError, ConnectionError, TimeoutError) as e:
                base, why = 30, f"{type(e).__name__} after up to {CFG['timeout']:g}s of silence"
            NOTE_HOOK and NOTE_HOOK(f"Ollama key #{i + 1}/{len(keys)}: {why}; parked for {_cool_down(keys[i], base):.0f}s, trying the next key")


@cache
def capabilities(model: str) -> tuple:
    """Capabilities Ollama reports for a model, e.g. ('completion', 'tools', 'thinking'); () if unknown."""
    try:
        return tuple(ollama.Client().show(model).capabilities or ())
    except Exception:
        return ()


def over_budget() -> bool:
    """True once the current phase (one condition's run) has spent its token budget (CFG token_budget, 0 = unlimited)."""
    return bool(CFG["token_budget"]) and USAGE["tokens"] >= CFG["token_budget"]


def make_llm(model: str, role: str, think: bool | None = None) -> ChatOllama:
    """`role` (agent_call | trust_audit | corruption) goes into metadata so the trace recorder can label the call."""
    common = dict(temperature=CFG["temperature"], seed=CFG["seed"], num_ctx=CFG["num_ctx"], num_predict=CFG["max_tokens"], metadata={"role": role})
    if CFG["cloud"]:
        return CloudChatOllama(model=CLOUD_MODELS[model], base_url=CLOUD_HOST, reasoning=CFG["think"] if think is None else think, **common)
    caps = capabilities(model)
    think = (CFG["think"] if think is None else think) and ("thinking" in caps or not caps)  # models without thinking support reject think=True
    llm = GuardedChatOllama(model=model, reasoning=think, client_kwargs={"timeout": CFG["timeout"]}, **common)
    llm._client = WatchedClient(timeout=CFG["timeout"])  # host from OLLAMA_HOST, like the default client
    return llm


# ---------------------------------------------------------------------------------------------- tools
_repeats: dict[tuple, int] = {}


def reset_repeats() -> None:
    """Forget which tool calls were made (called at the start of every agent turn)."""
    _repeats.clear()


def guarded(fn):
    """Tool wrapper: runs the call in a thread and gives up after TOOL_TIMEOUT s, returning an ERROR text the agent can react
    to (before, one hung search / download / `input()` / endless loop froze the whole run silently). An abandoned call is
    left behind; if it was the Python shell, its stdout capture is undone so the program's own output keeps working."""
    @functools.wraps(fn)
    def run(*a, **kw):
        res, stdout, t0 = {}, sys.stdout, time.time()
        arg = str((list(kw.values()) + list(a) or [""])[0]).replace("\n", " ")[:80]
        key = (fn.__name__, str(a), str(sorted(kw.items())))
        _repeats[key] = _repeats.get(key, 0) + 1
        if _repeats[key] > 2:  # models sometimes repeat one identical call 15 times in a row: stop it at the third
            return (f"ERROR: you already made this exact {fn.__name__} call {_repeats[key] - 1} times in this turn with the same result. "
                    "Repeating it is pointless: use a different approach, or give your best FINAL ANSWER now.")
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


SEARCH_DEAD = "Tavily keys are exhausted or rejected"  # also the marker used to find results made without search


@tool
@guarded
def web_search(query: str) -> str:
    """Search the web. Returns titles, URLs and snippets; open a promising page afterwards with read_file(url)."""
    global _key
    keys, codes = tavily_keys(), []
    for attempt in range(2):
        for _ in range(max(len(keys), 1)):
            key = keys[_key % len(keys)] if keys else ""
            try:  # plain REST call with a timeout (the langchain wrapper had none)
                r = requests.post("https://api.tavily.com/search", json={"query": query, "max_results": 5},
                                  headers={"Authorization": f"Bearer {key}"}, timeout=30)
            except requests.RequestException as e:  # a failed search must not crash the debate: the agent sees the error
                return f"ERROR: web search failed: {type(e).__name__}: {e}"
            if r.status_code in TAVILY_SKIP_CODES:  # key exhausted / rejected / rate limited -> next key
                codes.append(r.status_code)
                _key += 1
                continue
            if r.status_code != 200:
                return f"ERROR: web search failed: HTTP {r.status_code} {r.text[:200]}"
            hits = r.json().get("results") or []
            return ("\n\n".join(f"{h['title']} ({h['url']})\n{h['content']}" for h in hits) or "(no results)")[:MAX_CHARS]
        if attempt == 0 and codes and set(codes) == {429}:  # only rate limits: wait a moment, go round once more
            time.sleep(5)
            codes = []
            continue
        break
    # Every key failed. Do NOT hand the agent an error to work around (everything after this would be answered without web
    # access and silently corrupt the experiment): stop the run; it resumes from the checkpoint once keys are available.
    raise KeysExhausted(f"all {len(keys)} {SEARCH_DEAD} (HTTP {sorted(set(codes)) or 'no key'}); add Tavily keys or wait for the "
                        "quota reset, then rerun the same command (unfinished work resumes)")


SYSTEM = """You are a careful research assistant solving a GAIA benchmark question, possibly as one member of a team that debates the answer.
Work in several steps: plan, call tools, read their results, and keep going until you are sure. Tools:
- web_search(query): find facts; then open a page with read_file(url).
- read_file(path_or_url, offset=0): read an attachment (pdf, docx, xlsx, pptx, csv, json, zip, html...) or a web page as markdown, one page at a time; long documents continue at the offset the page names.
- Python_REPL: run Python for any calculation or data processing (state persists between calls; always print() what you want to see). Never do arithmetic in your head.
You have at most {STEPS} tool calls per turn: stop researching and answer before they run out. Older tool outputs are shortened to a preview to save space, so write down the facts you need as you find them. If a tool returns an error, fix the call or try another approach.
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
    return create_agent(make_llm(model, "agent_call"), [web_search, read_file, python_tool()], system_prompt=SYSTEM.replace("{STEPS}", str(CFG["max_steps"])),
                        middleware=[ContextEditingMiddleware(edits=[MaskOldObservations(budget=int(CFG["num_ctx"] * 1.2))])])
