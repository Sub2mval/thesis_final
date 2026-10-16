"""GAIA validation set + the official GAIA scorer.

The dataset is downloaded ONCE into ./data/gaia (override with the GAIA_DIR env var). Every later run finds the
`.download_complete` marker and reads everything locally: no network and no Hugging Face token needed.
(The HF account must have accepted the terms at https://huggingface.co/datasets/gaia-benchmark/GAIA.)
"""

import json
import os
import re
import string
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
GAIA_DIR = Path(os.getenv("GAIA_DIR", HERE / "data" / "gaia"))
VAL_DIR = GAIA_DIR / "2023" / "validation"
DONE = GAIA_DIR / ".download_complete"  # written only after a FULL download, so an interrupted one is redone


def ensure_gaia(token: Optional[str] = None) -> bool:
    """Make sure the validation split (metadata + attachments) is on disk. Returns True if it downloaded now."""
    if DONE.exists():
        return False
    if not token:
        raise SystemExit("GAIA is not downloaded yet and there is no HF_TOKEN in .env (needed for this one-time download).")
    from huggingface_hub import snapshot_download

    print(f"Downloading GAIA validation split once into {GAIA_DIR} ...", flush=True)
    snapshot_download("gaia-benchmark/GAIA", repo_type="dataset", local_dir=GAIA_DIR,
                      allow_patterns=["2023/validation/*"], token=token)
    DONE.write_text(datetime.now().isoformat())
    return True


def load_tasks() -> dict[str, dict]:
    """task_id -> {task_id, Question, Level, Final answer, file (absolute local path or None)}."""
    jsonl, parquet = VAL_DIR / "metadata.jsonl", VAL_DIR / "metadata.parquet"
    if jsonl.exists():
        rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:  # parquet needs pandas + pyarrow
        import pandas as pd

        rows = pd.read_parquet(parquet).to_dict("records")
    tasks = {}
    for r in rows:
        if r["task_id"] == "0-0-0-0-0" or not r.get("Question"):  # placeholder row
            continue
        name = r.get("file_name") or ""
        tasks[r["task_id"]] = {"task_id": r["task_id"], "Question": r["Question"], "Level": str(r["Level"]),
                               "Final answer": str(r["Final answer"]), "file": str(VAL_DIR / name) if name else None}
    return tasks


# --------------------------------------------------------------------------------------------------
# Official GAIA quasi-exact-match scorer (unchanged), used for the FINAL answer and for every STEP.
# --------------------------------------------------------------------------------------------------
def _is_float(x: Any) -> bool:
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


def _norm_number(s: str) -> float:
    for ch in ("$", "%", ","):
        s = s.replace(ch, "")
    try:
        return float(s)
    except ValueError:
        return float("inf")


def _norm_str(s: str, remove_punct: bool = True) -> str:
    s = re.sub(r"\s", "", s).lower()
    return s.translate(str.maketrans("", "", string.punctuation)) if remove_punct else s


def gaia_score(model_answer: Optional[str], truth: str) -> bool:
    model_answer = "None" if model_answer is None else str(model_answer)
    truth = str(truth)
    if _is_float(truth):
        return _norm_number(model_answer) == float(truth)
    if any(c in truth for c in ",;"):
        gt, ma = re.split("[,;]", truth), re.split("[,;]", model_answer)
        if len(gt) != len(ma):
            return False
        return all(
            (_norm_number(m) == float(g)) if _is_float(g) else (_norm_str(m, False) == _norm_str(g, False))
            for m, g in zip(ma, gt)
        )
    return _norm_str(model_answer) == _norm_str(truth)


def extract_answer(text: Optional[str]) -> str:
    """Text after the last `FINAL ANSWER:`, first line only, without markdown emphasis (`**8**` -> `8`).
    Local models like to bold or add trailing remarks; the plain official scorer would mark those wrong."""
    ans = (text or "").split("FINAL ANSWER:")[-1].strip()
    return (ans.splitlines()[0] if ans else "").strip().strip("*` ")
