# Phase 7 report: Streamlit demo

**Status:** Code-complete on synthetic fixtures; not verified. The app is a
software demonstration only. Real results remain **NOT YET RUN**; no labels were
created.

## Files built

- `app.py`: Streamlit launch entry point.
- `src/geointel/app/streamlit_demo.py`: fixture-only UI, synthetic metric cards,
  deterministic MockPlanner interaction, and a simple fixture comparison chart.
- `src/geointel/agent/planner.py`: deterministic mock planner used by the UI.
- `src/geointel/benchmark/phase6.py`: synthetic fixture values shown by the UI.
- `docs/demo_script.md`: five-minute presentation walkthrough.
- `docs/architecture.md`: system flow diagram.
- `docs/limitations.md`: data, validation, and demo constraints.
- `docs/phase7_report.md`: this status and screenshot checklist.

## Run command

From the project root, install the project dependencies if needed, then run:

```powershell
python -m pip install -e .
python -m streamlit run app.py
```

The UI displays **DEMO DATA — SYNTHETIC FIXTURE VALUES; NOT REAL GEO-INTEL
RESULTS** on the page and sidebar. It uses no real composites, labels, or live
planner calls.

## Screenshot checklist

Capture screenshots only after launching the app. Check that:

- [ ] The top warning banner is visible without scrolling.
- [ ] The sidebar repeats the full DEMO DATA warning.
- [ ] Each displayed metric card is marked `DEMO DATA`.
- [ ] The mock planner section states that its output is a demo and shows a
      refusal for a causal-trap query (for example, “Did urban expansion cause
      vegetation loss?”).
- [ ] The comparison chart caption identifies its data as synthetic fixture
      values.
- [ ] No screenshot is captioned or presented as a real GEO-INTEL result.

## Known limits

- The display is backed exclusively by hard-coded synthetic fixtures; it does
  not read raster outputs, run analysis, or represent Dehradun measurements.
- The planner is deterministic and mocked. The live LLM benchmark remains
  **NOT YET MEASURED** and is not part of the app.
- A Streamlit runtime launch and browser review have not been recorded as
  verified. The checklist above is a manual review task.
- This UI does not provide authentication, persistence, deployment, or a
  production data-loading workflow.
- **Real results: NOT YET RUN.** No labels were created.
