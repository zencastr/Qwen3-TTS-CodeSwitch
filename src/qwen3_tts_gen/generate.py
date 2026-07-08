"""Batch speech generation from a manifest.

Reads a manifest, (optionally) shards it for SLURM job arrays, loads a
Qwen3-TTS model once, synthesizes each item, writes a WAV per item, and emits a
JSONL report. One bad item is logged and skipped rather than killing the run.

Examples:
  python -m qwen3_tts_gen.generate manifests/example.jsonl --out-dir outputs/demo
  python -m qwen3_tts_gen.generate big.jsonl --num-shards 8 --shard-index 0 --skip-existing
  python -m qwen3_tts_gen.generate big.jsonl --dry-run        # validate, don't load model
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import manifest as M
from .config import RuntimeConfig, configure_hf_cache
from .models import DEFAULT_MODEL


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="qwen3-tts-generate",
        description="Generate speech for every item in a manifest (JSONL/CSV).",
    )
    ap.add_argument("manifest", help="path to a .jsonl/.ndjson/.json/.csv/.tsv manifest")
    ap.add_argument("--out-dir", default="outputs", help="output directory (default: outputs)")
    ap.add_argument("--audio-root", default=None,
                    help="resolve relative ref_audio paths against this directory")
    ap.add_argument("--model", default=None,
                    help=f"model alias or HF repo id (default: ${{QWEN3_TTS_MODEL}} or {DEFAULT_MODEL})")
    ap.add_argument("--language", default="English",
                    help="default language for items that don't set one (default: English)")
    ap.add_argument("--mode", default="clone", choices=M.VALID_MODES,
                    help="default mode for items that don't set one (default: clone)")
    ap.add_argument("--device", default=None, help="torch device (default: cuda:0)")
    ap.add_argument("--dtype", default=None, help="bfloat16 | float16 | float32")
    ap.add_argument("--attn", default=None,
                    help="attention impl: sdpa | flash_attention_2 | eager")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="items per generate() call (>1 is best-effort; falls back to "
                         "per-item on error). Default: 1")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="total shards (job-array size); shard with --shard-index")
    ap.add_argument("--shard-index", type=int, default=0, help="this shard's index [0, num-shards)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip items whose output WAV already exists (resume-friendly)")
    ap.add_argument("--limit", type=int, default=None, help="process at most N items (after sharding)")
    ap.add_argument("--report", default=None,
                    help="JSONL report path (default: <out-dir>/report[.shardN].jsonl)")
    ap.add_argument("--cache", default=None, help="override model cache dir (QWEN3_TTS_CACHE)")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="cap generated tokens per item; bounds worst-case runaway decode "
                         "(see qwen3-tts-auto-language-instability memory)")
    ap.add_argument("--repetition-penalty", type=float, default=None,
                    help="e.g. 1.3; mitigates non-terminating repetition loops, esp. with "
                         "language=auto on code-switched text")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate + print the plan without loading the model")
    return ap


def _select_items(args) -> list[M.Item]:
    items = M.load_manifest(
        args.manifest, default_language=args.language, default_mode=args.mode
    )
    M.resolve_paths(items, args.out_dir, args.audio_root)
    items = M.shard(items, args.num_shards, args.shard_index)
    if args.skip_existing:
        before = len(items)
        items = M.filter_existing(items)
        if before != len(items):
            print(f"[skip-existing] {before - len(items)}/{before} outputs already exist")
    if args.limit is not None:
        items = items[: args.limit]
    return items


def _report_path(args) -> Path:
    if args.report:
        return Path(args.report)
    suffix = f".shard{args.shard_index}" if args.num_shards > 1 else ""
    return Path(args.out_dir) / f"report{suffix}.jsonl"


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    configure_hf_cache(args.cache)

    items = _select_items(args)
    plan = (
        f"manifest={args.manifest} items={len(items)} "
        f"shard={args.shard_index}/{args.num_shards} out-dir={args.out_dir}"
    )
    print(f"[plan] {plan}")

    if args.dry_run:
        missing = M.missing_local_refs(items)
        for it in items[:10]:
            print(f"  {it.id}  mode={it.mode}  lang={it.language}  -> {it.out_path}")
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more")
        if missing:
            print(f"[warn] {len(missing)} item(s) reference a missing local ref_audio, e.g. "
                  f"{missing[0].id} -> {missing[0].ref_audio}")
        print("[dry-run] manifest is valid; model not loaded.")
        return 0

    if not items:
        print("[done] nothing to do (all outputs exist or manifest empty for this shard).")
        return 0

    cfg = RuntimeConfig.from_env(
        model=args.model, device=args.device, dtype=args.dtype, attn=args.attn
    )
    print(f"[model] {cfg.model_id} device={cfg.device} dtype={cfg.dtype} "
          f"attn={cfg.attn_implementation}")

    from . import backend  # heavy import deferred until we actually generate

    t_load = time.perf_counter()
    model = backend.load_model(cfg)
    print(f"[model] loaded in {time.perf_counter() - t_load:.1f}s")

    report_path = _report_path(args)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_err = 0
    total_audio = total_gen = 0.0
    t_start = time.perf_counter()

    with report_path.open("w", encoding="utf-8") as report:
        done = 0
        gen_kwargs = {}
        if args.max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = args.max_new_tokens
        if args.repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = args.repetition_penalty

        for chunk in _chunks(items, max(1, args.batch_size)):
            results = _run_chunk(backend, model, chunk, gen_kwargs)
            for it, (status, err, audio_s, gen_s) in zip(chunk, results):
                done += 1
                if status == "ok":
                    n_ok += 1
                    total_audio += audio_s
                    total_gen += gen_s
                else:
                    n_err += 1
                    print(f"[error] {it.id}: {err}", file=sys.stderr)
                report.write(json.dumps({
                    "id": it.id, "mode": it.mode, "language": it.language,
                    "status": status, "error": err,
                    "out": str(it.out_path), "ref_audio": it.ref_audio,
                    "audio_seconds": round(audio_s, 3), "gen_seconds": round(gen_s, 3),
                }) + "\n")
                report.flush()
                if done % 25 == 0 or done == len(items):
                    print(f"[progress] {done}/{len(items)}  ok={n_ok} err={n_err}")

    wall = time.perf_counter() - t_start
    rtf = (total_gen / total_audio) if total_audio else float("nan")
    print(
        f"[done] ok={n_ok} err={n_err} | audio={total_audio:.1f}s "
        f"gen={total_gen:.1f}s wall={wall:.1f}s | RTF={rtf:.3f} | report={report_path}"
    )
    return 1 if (n_ok == 0 and n_err > 0) else 0


def _run_chunk(backend, model, chunk, gen_kwargs=None):
    """Synthesize a chunk. Try a batched call for size>1, fall back per-item."""
    if len(chunk) == 1:
        return [_run_one(backend, model, chunk[0], gen_kwargs)]
    try:
        return _run_batched(backend, model, chunk, gen_kwargs)
    except Exception as exc:  # noqa: BLE001 - batching is best-effort
        print(f"[batch] batched call failed ({exc}); falling back to per-item",
              file=sys.stderr)
        return [_run_one(backend, model, it, gen_kwargs) for it in chunk]


def _run_one(backend, model, it, gen_kwargs=None):
    t0 = time.perf_counter()
    try:
        wav, sr = backend.synthesize(model, it, gen_kwargs)
        audio_s = backend.write_wav(it.out_path, wav, sr)
        return ("ok", None, audio_s, time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001 - isolate per-item failures
        return ("error", repr(exc), 0.0, time.perf_counter() - t0)


def _run_batched(backend, model, chunk, gen_kwargs=None):
    """Batched generation: pass parallel lists to a single generate_* call.

    Only attempted for homogeneous clone-mode chunks; anything else raises so
    the caller falls back to per-item.
    """
    if any(it.mode != "clone" for it in chunk):
        raise ValueError("batching only supported for clone mode")
    method = getattr(model, "generate_voice_clone")
    t0 = time.perf_counter()
    wavs, sr = method(
        text=[it.text for it in chunk],
        language=[it.language for it in chunk],
        ref_audio=[it.ref_audio for it in chunk],
        ref_text=[it.ref_text for it in chunk],
        **(gen_kwargs or {}),
    )
    gen_s = (time.perf_counter() - t0) / len(chunk)
    if not isinstance(wavs, (list, tuple)) or len(wavs) != len(chunk):
        raise ValueError(f"unexpected batched output ({type(wavs)}, len mismatch)")
    out = []
    for it, wav in zip(chunk, wavs):
        mono = backend._to_mono_float(wav)
        audio_s = backend.write_wav(it.out_path, mono, sr)
        out.append(("ok", None, audio_s, gen_s))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
