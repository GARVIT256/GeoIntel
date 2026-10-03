# Phase 6 report: agent, RAG, and output validation

**Status:** Code-complete on fixtures; not verified. This does not verify a
deployed LLM, real-data answer, or hosted runtime. All real-data results remain
**NOT YET RUN**. No labels were created.

## Files built

- `src/geointel/agent/planner.py`: deterministic `MockPlanner` for supported
  metric extraction and refusal of causal, forecasting, or unsupported asks.
- `src/geointel/agent/validator.py`: narrow JSON plan allowlist, causal-language
  linter, and unit-aware numeric provenance check that reports JSON paths.
- `src/geointel/agent/service.py`: validated plan → fixture retrieval → supplied
  result JSON response flow. It does not calculate metrics or invent missing
  values.
- `src/geointel/rag/retriever.py`: in-memory TF-IDF retriever with stable
  citation chunk IDs; Phase 6 benchmark corpus is fixture text only.
- `src/geointel/benchmark/phase6.py`, `scripts/phase6_benchmark.py`: 10-query
  benchmark and JSON/stdout output. Fixture values are synthetic test inputs,
  not Dehradun estimates.
- `tests/test_phase6_agent.py`: planner parsing, refusals, plan allowlist,
  provenance, causal lint, citations, and the complete fixture benchmark.
- `benchmarks/phase6/queries.json`, `gold_plans.json`: 40 user queries with
  gold plans stored separately from planner prompts.
- `src/geointel/agent/live_planner.py`, `scripts/phase6_live_benchmark.py`:
  opt-in provider planner and benchmark against the separate gold plans.
- `tests/test_phase6_adversarial.py`, `tests/test_phase6_live_benchmark.py`:
  adversarial validator and benchmark-separation cases.
- `docs/phase6_report.md`: implementation, tested fixture report, and limits.

## Ten-query fixture benchmark

Run:

```powershell
.\.venv\Scripts\python.exe scripts/phase6_benchmark.py
```

The mock planner is scored against its own cached expected plans. This checks
plumbing only; these values are not model accuracy, deployed-LLM quality, or
real-data findings.

| Fixture metric | Result |
|---|---:|
| Mock cached-plan plumbing self-match rate | 1.00 |
| Correct refusal rate (2 out-of-scope queries) | 1.00 |
| Plan validity rate | 1.00 |
| Citation rate on supported queries | 1.00 |
| Number-provenance pass rate | 1.00 |
| Causal-linter pass rate | 1.00 |

The benchmark queries ask for urban expansion area, vegetation-loss area,
Spearman rho, Moran's I, Gi* hotspot count, NDVI decline area, T1 valid-pixel
coverage, and tree-to-built transition area; two additional queries request
causal attribution and a flood forecast and must be refused. Values used to
exercise answer rendering are explicitly synthetic and appear only in
`FIXTURE_RESULTS`.

## Fixture tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_phase5_change.py tests/test_phase6_agent.py -q --tb=short
```

Actual earlier combined Phase 5/6 fixture result: **15 passed**, Python 3.14.2,
pytest 8.3.2. The run emitted pytest-asyncio deprecation warnings. The benchmark
script also ran successfully on fixtures. No hosted or real-data pipeline was
run.

An additional targeted Phase 6 suite after the current changes collected 16
tests and passed all 16 (627 dependency deprecation warnings).

## Live planner benchmark

The live planner receives the query and system schema only. Gold plans are read
by the scoring script after inference and are never included in provider prompts.
Choose a provider with `GEOINTEL_LLM_PROVIDER=openai` or `anthropic`; temperature
is fixed at 0. The script makes no calls unless explicitly invoked with
`--execute-live`. Its no-call run reported 40 queries available and
**NOT YET RUN / NOT YET MEASURED** because no provider was configured. Live scores
are therefore not available.

The adversarial tests rejected a rounded value not present in results, a hectare
value reported as km², a percentage inferred from a fraction, and an unsupported
year. They accepted exact percentage/year provenance. The causal linter rejected
“due to”, “because of”, and “responsible for”; a co-occurrence-only statement
passed. These are fixture validator checks, not live-planner benchmark outcomes.

## Real-data status and limits

- Real Phase 4–5 outputs, labels, accuracy, association results, and external
  checks: **NOT YET RUN**.
- Real-data agent answers and evidence citations: **NOT YET RUN**.
- Hosted Python 3.11/Colab execution: **NOT YET RUN**.
- This planner is a mock parser, not a connected language model. The benchmark
  evaluates only its declared ten fixtures; scores cannot be generalized to
  open-ended questions or an LLM.
- The number checker verifies exact numeric tokens and stated units against
  supplied JSON paths. It does not validate derived arithmetic, statistical
  interpretation, or whether source data are trustworthy.
- RAG citations point to retrieved chunks but do not establish that a claim is
  entailed by a chunk. Human review remains necessary.
- The causal linter is phrase-based and can miss novel causal wording or flag a
  phrase even when it appears in a negated sentence.

## Report text

Phase 6 adds a deterministic mock planner, strict plan allowlist, in-memory
fixture RAG retrieval, unit-aware JSON number-provenance checker, causal-language
linter, and separate opt-in live-planner benchmark. The mock's 1.00 scores test
cached-plan plumbing only; the real LLM planner is **NOT YET MEASURED**. Real
agent answers and raster outputs remain **NOT YET RUN**. No labels were created.
