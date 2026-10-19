# LLM-debate + trust allocator (local Ollama) on GAIA

```
pip install -r requirements.txt            # + a running Ollama with the models pulled (ollama pull <tag>)
cp .env.example .env                       # TAVILY_API_KEY (web search); HF_TOKEN only for the one-time GAIA download
python run_gaia_benchmark.py --n-questions 5 --agent-models gemma4:31b --trust-model qwen3.8:27b
python run_gaia_benchmark.py --task-ids <id> --agent-models gemma4:31b gemma4:12b qwen3.5:9b --trust-model muse-glimmer:30b
python analyze.py results/                 # slides 7-9 -> results/analysis/{report.md,runs.csv,steps.csv}
python -m pytest tests -q                  # offline tests against a fake Ollama (tests/mock_ollama.py)
```

| file | role |
|---|---|
| `gaia_data.py` | GAIA downloaded once into `data/gaia` (marker `.download_complete`), read locally afterwards; official scorer |
| `agents.py` | local Ollama LLM factory, tools (web search, read_file, Python), tool-using agent; `TRUST_MODELS` / `AGENT_MODELS` tags live here |
| `debate.py` | the LangGraph debate (3 agents, aggregator, optional trust node), trust audit, trust notice |
| `Gricean_check.py` | the Gricean rubric prompt (verbatim) + trust score = mean of the 4 maxim scores |
| `error_injection.py` | FM-2.1..2.6 only; the injector picks the best-fitting mode for message 1 and corrupts it once per task (shared by all injected runs) |
| `run_gaia_benchmark.py` | runs the 4 conditions per task; records every LLM call; writes `.json`, `.md`, `.full.md`, `summary.csv` |
| `analyze.py` | analysis per the slides |

**Models.** Trust allocator: `muse-glimmer:30b | gemma4:31b | qwen3.8:27b`. Agents: those + `gemma4:12b | qwen3.5:9b`
(one tag = all three agents, or three tags = agent 1/2/3; the aggregator uses agent 1's model). Independent of each other.
The injector model is `--injector-model` (default `gemma4:31b`). Startup checks that every model is pulled and agents can call tools.
Thinking is switched off automatically for models without it (`--no-think` forces it). Context window `--num-ctx` (default 32768); tool outputs are capped at 15k chars.

**What is judged.** Every public agent message ends in `FINAL ANSWER: ...`; it is scored with the GAIA scorer exactly like the final
answer (`step_correct`). Trust score = mean of the four 1-5 maxim scores; agents see only that score and the auditor's reasons.

**Analysis definitions.** Error propagation = number of inaccurate steps in a row counted from the injected step (0 if the injected step
is still accurate; "never recovered" = still wrong at the last step). RQ: fork_off vs fork_on, paired by task (McNemar, Wilcoxon).
VQ1: AUC / Mann-Whitney / Spearman of trust vs step accuracy on clean allocator-on runs (task-bootstrap CI). VQ2: trust of the injected vs the
original message (paired Wilcoxon). Exploratory: "mistake" = trust >= `--high` (4.5) on an inaccurate step or < `--low` (3.0) on an accurate one.

Layout per task: `baseline_off`, `baseline_on` (clean), `fork_off`, `fork_on` (message 1 replaced by its corrupted copy).
Message 1 (Agent 1's first answer) and its corruption are generated ONCE per task and shared by all four conditions, so the injected
message is byte-identical with and without the allocator (`shared` calls are listed first in each `.full.md`). If the injector
deems message 1 ineligible, the `fork_*` records repeat the clean run and are flagged `injection.eligible=false` (analysis skips them).

**Context handling.** Tool outputs fill the window. `read_file` returns 8k-char pages with an explicit `offset=` footer (the full
document is converted once and cached); the newest tool outputs stay in full (~1/3 of `--num-ctx`), older ones shrink to a 300-char
preview marked as hidden. The agent's own reasoning/actions are never touched, and the stored trace stays complete. If an answer
is still cut off, the agent is asked once for a best-effort final answer.

**Context window / null answers (fixed after the first Colab sample).** Tool outputs filled the 32k window; Ollama then cut generation
mid-thinking and the answer came back empty (`(no answer produced)`). Now (1) all but the 3 newest tool outputs are hidden from the model once
the history passes num_ctx/2 (full history is still stored), (2) the Python tool's output is capped like the others, (3) a message that ends
without `FINAL ANSWER:` is retried once with a "give your best answer now" nudge, (4) every call records `done_reason` and logs a warning when it is
`length`. `analyze.py` reports steps without a FINAL ANSWER separately and excludes them from VQ1.

**What the trust allocator sees.** Exactly what the audited agent could see, and nothing more: the task, the public answers so far,
the attachment, and the audited agent's OWN tool calls and raw tool outputs, in a block labelled `[TOOL OUTPUTS SEEN BY Agent N]`
(shortened the same way the agent's own view was). Never its reasoning, never another agent's tool results. For an injected message 1
the auditor sees Agent 1's genuine tool outputs next to the corrupted text (the same in every injected run).
