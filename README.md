# Bonsai-2-27B evaluation pipeline (v2)

A resource-aware, sharded benchmark for **Ternary-Bonsai-2-27B** running locally under `llama-server` on free GitHub-hosted runners.
Stdlib-only Python (Pillow only for the vision shard, matplotlib optional for charts). No paid APIs, no model files in Git.

## Run it

**Actions → "Bonsai-2-27B evaluation (matrix)" → Run workflow.** Defaults run everything (~160 tests) in 14 parallel jobs.

| input | meaning |
|---|---|
| `suites` | `all` or e.g. `coding,agentic` |
| `repeat`, `seed`, `temperature`, `top_p`, `thinking`, `max_tokens`, `context_size` | sampling / runtime controls (all recorded in every shard's `run_config.json`) |
| `parallel_jobs`, `shard_minutes` | matrix `max-parallel` and the target length of a shard |
| `stress_level` | `low` / `medium` / `high` (the kill-and-restart test needs ≥ medium) |
| `enable_vision`, `enable_long_context`, `long_context_levels` | toggle expensive suites; levels from `1k,2k,4k,8k` |
| `resource_policy` | `strict` aborts a shard that lacks RAM; `permissive` only warns |
| `budget_minutes` | per-shard hard budget; remaining tests become **skipped**, not failed |
| `test_filter` | regex on test ids, to re-run only some tests (e.g. `^think-1`) |
| `publish` | commit compact results to the `benchmark-results` branch |

Local, no model needed (verifies the harness itself): `python3 -m bench selftest --e2e`

## What is tested (167 test definitions)

| suite | tests | what it probes |
|---|---|---|
| basic | 18 | arithmetic, facts, strict formatting, JSON, tokenization trap (`strawberry`) |
| thinking | 22 | CRT puzzles, algebra, probability, combinatorics, knights/knaves, constraint puzzles (verified, not string-matched), Tuesday-boy, jugs, bridge crossing |
| coding | 18 | **executed** in a sandbox against hidden tests: debugging, edge cases, O(n) vs O(n²), LRU cache, CSV parser, DP, expression parser, and **unit-test generation graded by mutation** |
| agentic | 9 | native tool calls against real mock tools: selection, restraint, multi-step, state, file I/O, flaky-API retry, **prompt injection in a tool result**, asking for missing info, Python tool |
| advanced | 10 | JSON extraction, multi-constraint writing, calibration on unanswerable questions, table reasoning, cipher, code tracing, system-prompt adherence |
| consistency | 30 | 6 families × (original / paraphrase / reordered / irrelevant wording / repeat at T=0.7) → answer-agreement and accuracy |
| long_context | 15 (+5) | needles at start/middle/end + corrections + near-duplicate distractors at ~1k/2k/4k (8k opt-in) |
| vision | 10 | images **generated locally** with known ground truth: counting, colour, OCR, charts, table, spatial, equation, blank-image honesty |
| performance | 10 | cold start, warm latency, TTFT, output/prompt length scaling, prompt-cache reuse, concurrency ×2 |
| stress | 8 | burst, long generation, mixed sizes, concurrent burst, client abort, malformed requests, context overflow, `kill -9` recovery |
| safety | 12 | refusals, **over-refusal**, injection, secret keeping, instruction hierarchy, hallucination, adversarial formatting |

Every test has a difficulty `level` (1–4) and a `category`, and reports slice by both.

## Design decisions (grounded in the 2026-11-25 run)

* **Throughput was ~1.6 tok/s decode and ~4 tok/s prefill** on the 4-vCPU runner, so 27 tests took ~1 h. Everything is sized around that:
  per-test `max_tokens`, planning estimates (`bench/cases.py`), a per-shard time budget, and long-context questions that share one document so the prompt cache pays the prefill once.
* **Parallelism is across VMs, not inside one.** Each matrix job is its own runner, so jobs cannot starve each other of RAM. Inside a job the server runs `-np 1` (only `performance`/`stress` use `-np 2`, because they test concurrency) — extra slots on 4 cores would only split the same throughput.
* **`adv-05` was a grader false negative** (the model answered `\frac{3}{10}`). Graders now parse fractions/LaTeX/percent/thousands separators, and `python3 -m bench selftest` proves for every test that its reference answer passes and a plausible wrong answer fails. Run it after editing any test.
* **Vision failed because llama-server could not fetch a Wikimedia URL** (HTTP 500 in 0.1 s). Images are now generated locally and sent as base64; if the server reports no vision modality the suite is **skipped with a reason**, never scored as a model failure.
* Accuracy = passed / (passed + failed + truncated + timeout). Errors, grader errors and skips are listed separately. `truncated` (hit `max_tokens`) is its own status, because for a reasoning model "ran out of budget" differs from "wrong".

## Where everything is

Each shard artifact `shard-<name>/` contains: `transcript.md` (human-readable prompt → reasoning → response → checks for every test), `tests/<id>__r<n>.json` (prompt, raw request/response, reasoning, tool-call trace, every grading check, tokens, server timings, retry history, resources before/after), `results.jsonl`, `summary.{json,md}`, `metadata/` (run config, model + hashes, runtime version/command), `diagnostics/` (`preflight.txt`, `memory.jsonl` every 5 s, `server.log`, `events.log`, OOM evidence).
The final artifact `benchmark-run-<id>/` adds `reports/overall.{json,csv,md}`, per-suite summaries, charts and a `manifest.json` with SHA-256 of every file. The same Markdown lands in the Actions job summary.

Actions console: each test prints a one-line `PASS/FAIL/TRNC/TIME/ERR/SKIP` status and an expandable group with prompt, reasoning, response and grading checks.

## Failure handling

Test failure → recorded, run continues. HTTP/connect errors → retried with backoff; a dead server is restarted (max 2) with the exit code captured (`-9` ⇒ probable OOM kill). Hash mismatch → file rejected, job fails. Insufficient disk → abort *before* download. Server won't start → exit code 3, every test recorded as `error/infrastructure`, logs preserved. Heavy tests are skipped if `MemAvailable` falls below a floor. A shard that never reports is flagged **missing** in the report.

## Adding tests

Edit `tests/<suite>.json`: `id`, `level`, `category`, `prompt`, `grade`, `max_tokens`, `est_out_tokens`, plus `reference` and `wrong` so selftest can validate the grader. Grader types: `numeric exact choice number_list lines contains_all contains_any not_contains regex json constraints order refusal uncertainty code unit_tests trace` (see `bench/graders.py`). Bump `DATASET_VERSION` in `bench/__init__.py` when you change tests, so runs stay comparable.

## Known limits (be aware)

* `refusal` / `uncertainty` graders are **keyword heuristics** (flagged `heuristic` in results); spot-check safety in `transcript.md`.
* Sandbox: separate process, scrubbed env, rlimits, timeouts, no network (`unshare -rn` when the kernel allows it, in-process guards otherwise; the mode used is recorded). It is best-effort, not a security boundary; eval jobs also hold a read-only token and no secrets.
* Hosted-runner speed varies; treat small latency/throughput differences as noise.
* The planner's time estimates use the 1.6 / 4 tok/s measured once; actual prefill at 2k–4k tokens is unmeasured until the `performance` shard reports it.
