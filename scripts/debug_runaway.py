#!/usr/bin/env python3
"""Diagnose the non-terminating/repetition-loop generation seen in the
CodemixQA clone run: 13/192 items in job 636 generated *exactly* 655.28s of
audio each, regardless of text content -- consistent with hitting a shared
default max_new_tokens ceiling rather than per-text repetition collapse.
Isolates root cause across a few axes, with a hard per-call wall-clock
timeout so one runaway condition can't eat the whole debug slot, and writes
results incrementally (flushed after every call) so a time-limit kill
doesn't lose progress like the first attempt did.

  - capped_600tok tested FIRST: confirms max_new_tokens actually bounds the
    worst case cleanly (the practical mitigation regardless of root cause)
  - language="auto" vs an explicit language name (does forcing the language
    avoid triggering the runaway at all?)
  - do_sample True/False, repetition_penalty
  - raw text vs text with markdown artifacts (*, ") stripped

Run on the debug partition (30 min cap):
  sbatch --partition=debug --gres=gpu:1 --cpus-per-task=8 --mem=64G \
    --time=00:25:00 --wrap="cd ~/qwen3_tts && .venv/bin/python scripts/debug_runaway.py"
"""
from __future__ import annotations

import json
import re
import signal
import time
from pathlib import Path

BAD_IDS = [
    "66-grammarforce_target-Simplified_Chinese",
    "1225-force-Korean",
    "3328-grammarforce_target-Japanese",   # contains a refusal-style artifact
]

PER_CALL_TIMEOUT_S = 120

MANIFEST = Path("manifests/codemixqa_full.jsonl")
LANG_FROM_ID = {
    "Simplified_Chinese": "Chinese",
    "Japanese": "Japanese",
    "Korean": "Korean",
    "Spanish": "Spanish",
    "Italian": "Italian",
    "French": "French",
}


class CallTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise CallTimeout(f"exceeded {PER_CALL_TIMEOUT_S}s")


def load_items():
    by_id = {}
    with MANIFEST.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d["id"] in BAD_IDS:
                by_id[d["id"]] = d
    return [by_id[i] for i in BAD_IDS]


def strip_artifacts(text: str) -> str:
    return re.sub(r"[*\"]", "", text)


def main() -> int:
    from qwen3_tts_gen import backend
    from qwen3_tts_gen.config import RuntimeConfig

    cfg = RuntimeConfig.from_env()
    print(f"[model] loading {cfg.model_id} ...", flush=True)
    t0 = time.perf_counter()
    model = backend.load_model(cfg)
    print(f"[model] loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    items = load_items()
    lang_for = {it["id"]: LANG_FROM_ID[it["id"].rsplit("-", 1)[-1]] for it in items}

    # capped first: cheapest + most practically useful result if time runs out.
    conditions = [
        ("capped_600tok", dict(language="auto", max_new_tokens=600)),
        ("explicit_lang", dict(language=None)),          # filled per-item below
        ("greedy", dict(language="auto", do_sample=False)),
        ("rep_penalty_1.3", dict(language="auto", repetition_penalty=1.3)),
        ("stripped_text", dict(language="auto")),          # text stripped below
        ("auto_default", dict(language="auto")),           # reproduce baseline last
    ]

    signal.signal(signal.SIGALRM, _alarm_handler)
    out_path = Path("logs/debug_runaway_results.jsonl")
    out_fh = out_path.open("w", encoding="utf-8")

    for it in items:
        for cond_name, extra in conditions:
            kwargs = dict(
                text=it["text"],
                ref_audio=it["ref_audio"],
                ref_text=it["ref_text"],
            )
            kwargs.update(extra)
            if cond_name == "explicit_lang":
                kwargs["language"] = lang_for[it["id"]]
            if cond_name == "stripped_text":
                kwargs["text"] = strip_artifacts(it["text"])

            t0 = time.perf_counter()
            signal.alarm(PER_CALL_TIMEOUT_S)
            try:
                wavs, sr = model.generate_voice_clone(**kwargs)
                wall = time.perf_counter() - t0
                audio_s = len(wavs[0]) / sr
                status, err = "ok", None
            except CallTimeout as exc:
                wall = time.perf_counter() - t0
                audio_s = None
                status, err = "timeout", repr(exc)
            except Exception as exc:  # noqa: BLE001
                wall = time.perf_counter() - t0
                audio_s = None
                status, err = "error", repr(exc)
            finally:
                signal.alarm(0)

            row = dict(id=it["id"], condition=cond_name, wall_s=round(wall, 1),
                       audio_s=(round(audio_s, 2) if audio_s is not None else None),
                       status=status, error=err)
            print(f"[result] {row}", flush=True)
            out_fh.write(json.dumps(row) + "\n")
            out_fh.flush()

    out_fh.close()
    print(f"[done] wrote results to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
