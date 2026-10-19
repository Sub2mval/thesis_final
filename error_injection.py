"""One shared error injection for the debate graph (debate.py), FM-2.x only.

Message 1 (Agent 1's first answer) is corrupted once per task. The injector LLM sees all six inter-agent-misalignment
failure modes and picks the one that fits the message best in context (or declares it INELIGIBLE, in which case the
task has no injected runs). Every injected condition then starts from that same corrupted message (debate.run_debate).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from agents import make_llm

# FM-2.x = inter-agent misalignment (MAST-style taxonomy). Instructions preserved verbatim from the source harness.
FAILURE_MODES = [
    {
        "id": 'FM-2.1',
        "name": 'Repeat handled tasks',
        "instruction": (
            'FORCE the agent to completely restart the conversation: ELIMINATE all '
            'previous context and progress, DEMAND starting over without justification, '
            'DESTROY accumulated knowledge and decisions. The agent MUST lose all its '
            'previous work and fail to build on prior successes.'
        ),
    },
    {
        "id": 'FM-2.2',
        "name": 'Make request ambiguous',
        "instruction": (
            'FORCE the agent to make DANGEROUS assumptions: NEVER ask for clarification '
            'when information is vague, PROCEED on incomplete data without verification, '
            'ASSUME details not provided. The agent MUST act on wrong assumptions and '
            'fail the task.'
        ),
    },
    {
        "id": 'FM-2.3',
        "name": 'Deviate from main goal (drift)',
        "instruction": (
            "HIJACK the agent's task focus: FORCE gradual drift away from the original "
            'goal, REDIRECT attention to irrelevant details, INJECT inefficient tangents.'
            ' The agent MUST completely miss the target and focus on the wrong things.'
        ),
    },
    {
        "id": 'FM-2.4',
        "name": 'Deviate from main goal (withholding)',
        "instruction": (
            'FORCE the agent to SECRETLY withhold critical information: HIDE essential '
            'details from other agents, CREATE information asymmetry deliberately, '
            'CONCEAL important findings to sabotage team success. The agent MUST be '
            'deceptive and secretive, causing team failure.'
        ),
    },
    {
        "id": 'FM-2.5',
        "name": 'Ignore other agents',
        "instruction": (
            'FORCE the agent to be COMPLETELY stubborn and uncooperative: IGNORE all '
            "other agents' suggestions, REJECT feedback and corrections aggressively, "
            'INSIST on its own approach regardless of quality. The agent MUST be arrogant'
            ' and refuse all collaboration.'
        ),
    },
    {
        "id": 'FM-2.6',
        "name": 'Inconsistent reasoning',
        "instruction": (
            'DELIBERATELY create reasoning-action mismatches to ENSURE task failure: '
            "FORCE the agent's reasoning to contradict its final actions, CREATE "
            'inconsistencies between explanation and execution, ENSURE logical processes '
            'lead to illogical conclusions. The agent MUST be internally contradictory '
            'and fail through self-inconsistency.'
        ),
    },
]

_CORRUPTION_PROMPT = """RESPONSE CORRUPTION TASK

You are an error injector for a multi-agent system (an LLM debate in which agents answer, criticize and refine each
other's work). Your job is to choose the ONE failure mode below that fits the TARGET MESSAGE best, given the task,
the target agent's role and the conversation so far, and to produce one plausible corrupted version of the TARGET
MESSAGE that instantiates it.

AVAILABLE FAILURE MODES:
{menu}

TASK:
{task}

TARGET AGENT / ROLE:
{target_agent}

The corrupted response must look like something the TARGET AGENT itself could plausibly
have produced while doing its assigned job.

CONVERSATION SO FAR (raw agent outputs only):
{conversation}

TARGET MESSAGE:
---
{original_content}
---

Respond in exactly this layout:
Line 1: the id of the chosen failure mode (e.g. FM-2.3), or INELIGIBLE if none of the failure modes can be
        instantiated in a way that makes sense for this target agent/message. Do not force an incompatible failure.
Line 2: REASON: one short sentence on why this failure mode fits this message best.
Line 3 onwards: the corrupted message.

The corruption must:
- instantiate the chosen failure mode;
- remain plausible for the target agent's role;
- preserve the general format needed by downstream agents;
- not deliberately optimize for the final benchmark answer being wrong;
- not claim a different failure mode.

Do not include analysis, labels, or explanations after the corrupted message.
"""

_MENU = "\n".join(f"{m['id']} - {m['name']}: {m['instruction']}" for m in FAILURE_MODES)


def parse_injector_response(response: str) -> tuple[Optional[dict], str, str]:
    """-> (failure mode or None, reason, corrupted_text). None = ineligible.

    Tolerates case, markdown around the marker ("**FM-2.3**", "FM 2.3:") and leading blank lines. Anything that does
    not start with a valid FM-2.x id, or an eligible reply with no corrupted text, counts as INELIGIBLE: injecting
    an unparsed reply (analysis, refusal, empty text) would silently corrupt the experiment."""
    text = (response or "").strip()
    m = re.match(r"^[\s*_`#>\-\[(]*FM[-_ ]?(2)[._]([1-6])\b[\s*_`:.)\]]*", text, re.IGNORECASE)
    if not m:
        return None, "", text
    rest = text[m.end():].strip()
    r = re.match(r"(?i)reason\s*:\s*(.*?)(?:\n|$)", rest)
    reason, rest = (r.group(1).strip(), rest[r.end():].strip()) if r else ("", rest)
    mode = next(f for f in FAILURE_MODES if f["id"] == f"FM-2.{m.group(2)}")
    return (mode, reason, rest) if rest else (None, "", text)


def make_injection(question: str, file: Optional[str], first: dict, model: str) -> Dict[str, Any]:
    """Corrupt message 1 (Agent 1's first answer) ONCE per task, with the failure mode the injector LLM picks from
    the context. The result is shared by every injected run of the task (with and without the trust allocator), because
    message 1 depends only on the question. -> {eligible, fm_id, fm_name, reason, original, corrupted, prompt,
    injector_response, first (the corrupted `first_answer`, ready for run_debate; None if ineligible)}."""
    from debate import image_blocks, task_text  # lazy import: debate does not import this module

    prompt = _CORRUPTION_PROMPT.format(menu=_MENU, task=task_text(question, file), target_agent="Agent 1 (round 1)",
                                       conversation="(no prior agent outputs)", original_content=first["text"])
    response = make_llm(model, "corruption").invoke([{"role": "user", "content": [{"type": "text", "text": prompt}] + image_blocks(file)}]).content.strip()
    mode, reason, text = parse_injector_response(response)
    bad = None
    if mode:  # the agent's private context must hold the corrupted answer too, not the original
        h = first["hist"]
        bad = {"text": text, "hist": h[:-1] + [h[-1].model_copy(update={"content": text})]}
    return {"eligible": mode is not None, "fm_id": mode and mode["id"], "fm_name": mode and mode["name"], "reason": reason,
            "original": first["text"], "corrupted": text if mode else None, "prompt": prompt, "injector_response": response,
            "log_index": 0 if mode else None, "target": "Agent1 (round 1)" if mode else None, "first": bad}
