# Bonsai 2 27B CI Capability Evaluation

This repository runs a collection of **lightweight, reproducible capability checks** against Bonsai 2 27B using a local `llama-server` OpenAI-compatible endpoint on GitHub-hosted runners.

> This is a screening/evaluation harness, not a validated benchmark. Most grading is heuristic (string/regex/structure checks); a pass is not a guarantee of correctness. For serious comparisons, use fixed prompts, multiple runs, blinded model comparison, executable hidden tests, and pre-registration.

## Pipelines

| Suite | Difficulty / focus | Examples |
|---|---|---|
| `basic` | beginner smoke tests | arithmetic, short factual answer, concise instruction adherence, JSON |
| `thinking` | reasoning | multi-step arithmetic, logic, constraints, planning, deduction |
| `coding` | code generation & debugging | Python functions, edge cases, binary search, explanation |
| `agentic` | tool use and planning | tool choice, call planning, JSON-shaped tool call, safe action planning |
| `advanced` | robustness / structured outputs | synthesis, JSON, prompt-injection resistance, uncertainty, probability |
| `vision` | multimodal image understanding | image description and grounded scene classification (requires image URL access) |
| `all` | run all suites | longer job; may exceed hosted runner time budget |

## Run it in GitHub Actions

1. Open **Actions → Bonsai 2 27B Capability Evaluation**.
2. Choose **Run workflow**.
3. Select a suite (start with `basic`, then `thinking` / `coding`, and later `agentic`, `advanced`, `vision`, or `all`).
4. Set repetitions and token limit. Repetition is useful for variability but increases run time.
5. Download the artifact for raw JSONL results, summary JSON, server logs, and model SHA-256 hashes.

The workflow is manual-dispatch by design to avoid downloading a multi-GB model on every commit. Runner CPU inference can be slow; set a sensible token limit and allow a long timeout.

## Run locally against an OpenAI-compatible endpoint

```bash
python3 scripts/evaluate.py --base-url http://127.0.0.1:8080/v1 --suites basic
python3 scripts/evaluate.py --suites thinking,coding --repeat 2 --max-tokens 4096 --thinking on
python3 scripts/evaluate.py --suites all --max-tokens 2048
```

Environment variable: `OPENAI_BASE_URL` (defaults to `http://127.0.0.1:8080/v1`).

## Artifacts / measurements

Each run writes:
- `results/run_<UTC timestamp>.jsonl`: one record per test, prompt response, reasoning field if provided by server, request latency, API usage, finish reason, check details and errors.
- `results/summary_<UTC timestamp>.json`: aggregate request counts, heuristic pass rate, mean latency and selected suites.
- Workflow artifact: raw result files, server logs, model hash file, runtime archive.

The tests record latency and reported token usage, but token throughput is only reported if the API returns usage counters. Hosted runner measurements are hardware-specific and should not be compared directly to local GPU results.

## Model/runtime configuration

The default downloads the `PQ2_0` GGUF and matching mmproj from PrismML's Hugging Face repository and uses PrismML's llama.cpp runtime. `--jinja` is enabled for chat-template/tool-format support. The workflow runs CPU-only (`-ngl 0`) on Ubuntu hosted runners and uses an 8 GB swap best-effort.

## Important caveats

- GitHub-hosted runners are not intended for sustained inference of large models. This may be slow, memory constrained, or fail. It is a CI smoke/evaluation environment, not a hardware performance benchmark.
- Test prompts and heuristic graders are visible in this repo; contamination is possible.
- `vision` requires the runner to reach the image URL and the model server to support image input with the projector. If it fails, inspect server logs. Do not interpret network failures as a model capability failure.
- Agentic suite evaluates planning and tool-call descriptions only. It does **not** execute tools in a live sandbox.
- The current coding checks are pattern-based. An advanced follow-up can add isolated Python execution, generated tests, SWE-bench-style repo tasks, and tool-round-trip environments.
- For meaningful results, keep model/runtime hashes, settings, prompt versions, and runner environment together; rerun each task several times if measuring stability.

## Folder layout

```text
.github/workflows/
  call_27b.yml       # manual entry workflow + reusable-call interface
  exec.yml           # reusable wrapper
scripts/
  evaluate.py        # endpoint client, graders, result/summary writer
tests/
  basic.json
  thinking.json
  coding.json
  agentic.json
  advanced.json
  vision.json
results/              # generated output, not committed
```

## Next recommended upgrades

1. Add executable code tests in a sandbox (Docker unavailable on hosted runner? use a subprocess with timeouts and restricted environment).
2. Add tool-call round-trip tests with a mock tool registry and state assertions.
3. Add long-context tests (gradually increase input length and check retrieval/needle accuracy).
4. Add metamorphic tests (equivalent prompts, arithmetic invariants, translation consistency).
5. Add paired-model evaluation and confidence intervals; freeze task sets before model comparison.
