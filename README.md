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
| `error_injection.py` | FM-2.1..2.6 only; the injector picks the best-fitting mode for the target message, forks the checkpoint, resumes |
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

Layout per task: `baseline_off`, `baseline_on` (clean), `fork_off`, `fork_on` (same clean run forked at `--message-index`, default 1, with an injected error).
