# Five-minute GEO-INTEL demo script

All values shown during this walkthrough are synthetic **DEMO DATA**. Do not
describe them as real study-area findings.

| Time | Walkthrough |
|---|---|
| 0:00–0:40 | Open the app with `python -m streamlit run app.py`. Point out the page and sidebar warnings: “DEMO DATA — SYNTHETIC FIXTURE VALUES; NOT REAL GEO-INTEL RESULTS.” State that this is a UI/plumbing demonstration. |
| 0:40–1:30 | Review the fixture metric cards. Explain that these are fixed test values used to demonstrate display formatting; they are not outputs from satellite imagery. |
| 1:30–2:20 | Submit “How much urban expansion in Dehradun?” in the mock planner. Show the structured plan and schema-valid indicator. Explain that the mock planner does not call an LLM or calculate area. |
| 2:20–3:15 | Submit the causal-trap query: “Did urban expansion cause vegetation loss?” Show the mock planner’s refusal. Explain the evidence policy: the project may describe spatial co-occurrence/association, but these analyses do not establish cause. |
| 3:15–4:05 | Review the comparison chart and call out its caption identifying synthetic fixture values. Avoid interpreting the visual as a temporal measurement. |
| 4:05–5:00 | Close with status: hosted Colab outputs, independent labels/accuracy, PostGIS checks, PDF verification, and reference-tile checks are outstanding. Point to `docs/STATUS.md` and `docs/limitations.md`. |

If the app does not show the warning banners or if any displayed value appears to
come from an external run, stop the demo and review the code before presenting.
