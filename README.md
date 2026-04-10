# MAST Meta-Observer for DeepAgents

Metacognitive monitoring wrapper for DeepAgents with trajectory-aware failure detection, checkpointing, and rollback.

## What this project does

- Wraps DeepAgents without forking its core implementation.
- Runs in two modes:
  - observer: MAST signatures + decisive rollback policy.
  - baseline: plain DeepAgents execution path.
- Stores generated code under `created-projects/<project-name>/runs/<run-id>/`.
- Stores observer artifacts under `created-projects/<project-name>/.mast/<run-id>/`.

## Install

```bash
uv sync
```

This project references the cloned DeepAgents source at `../deepagents/libs/deepagents`.

## Run

### Observer mode

```bash
python run.py "Build an LMS" --project lms --mode observer
```

### Baseline mode

```bash
python run.py "Build an LMS" --project lms --mode baseline
```

### Benchmark (observer vs baseline)

```bash
python benchmark.py "Build a todo app" --project bench-todo
```

## CLI options

```text
python run.py <task> [options]

  --project NAME      Folder name under created-projects/   [default: my-project]
  --mode              observer | baseline                    [default: baseline]
  --model             LangChain model string                 [default: openai:gpt-5.4-nano]
  --reasoning-effort  low | medium | high | none            [default: high]
  --window            Sliding window size                    [default: 10]
  --threshold         Failure score threshold (0.0-1.0)      [default: 0.75]
  --persistence       Consecutive windows before rollback    [default: 2]
```

## Artifact layout

```text
created-projects/
└── todo-app/
  ├── runs/
  │   └── 2026-04-10T07-00-00Z-acde1234/
  │       ├── app.py
  │       └── templates/
  └── .mast/
    └── 2026-04-10T07-00-00Z-acde1234/
      ├── trace.json
      ├── report.json
      └── checkpoints/
        └── index.json
```

## Implemented MAST signatures

1. Tool-use loop
2. Repeated invalid action
3. No progress or stagnation
4. Context overload
5. Malformed tool output
6. Role disobedience
7. Instruction drift

## Notes

- Rollback policy uses `threshold` and `persistence` to avoid false positives.
- Checkpoint handling currently uses LangGraph `MemorySaver` only.
- MTTR in `report.json` is measured as event-distance recovery proxy.
- Set `OPENAI_API_KEY` in `.env` to run against OpenAI; `ANTHROPIC_API_KEY` is no longer the default path.
- Single source of truth for runtime defaults is in `src/mast_meta_observer/config.py` (`DEFAULT_MODEL_SPEC`, `DEFAULT_REASONING_EFFORT`, `DEFAULT_WINDOW`, `DEFAULT_THRESHOLD`, `DEFAULT_PERSISTENCE`).
- Optional env overrides: `MAST_MODEL`, `MAST_REASONING_EFFORT`, `MAST_WINDOW`, `MAST_THRESHOLD`, and `MAST_PERSISTENCE`.
