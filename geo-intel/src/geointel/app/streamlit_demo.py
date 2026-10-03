"""Phase 7 Streamlit demonstration backed exclusively by synthetic fixtures."""
from __future__ import annotations

import streamlit as st

from geointel.agent.planner import MockPlanner
from geointel.agent.validator import validate_plan
from geointel.benchmark.phase6 import FIXTURE_RESULTS


DEMO_BANNER = "DEMO DATA — SYNTHETIC FIXTURE VALUES; NOT REAL GEO-INTEL RESULTS"


def main() -> None:
    st.set_page_config(page_title="GEO-INTEL Demo", layout="wide")
    st.error(DEMO_BANNER, icon="⚠️")
    st.title("GEO-INTEL | Phase 7 demonstration")
    st.warning("Every value and answer on this page is synthetic demo data.", icon="⚠️")
    st.sidebar.error(DEMO_BANNER)
    st.sidebar.caption("Fixture-only planner and fixture values; no real outputs are loaded.")

    st.subheader("Synthetic fixture values")
    columns = st.columns(3)
    for index, (key, value) in enumerate(FIXTURE_RESULTS.items()):
        unit = "ha" if key.endswith("_ha") else "%" if key.endswith("_pct") else ""
        columns[index % len(columns)].metric(f"DEMO DATA · {key}", f"{value}{unit}")

    st.subheader("Mock planner plumbing demo")
    query = st.text_input("Enter a supported demo question", "How much urban expansion in Dehradun?")
    if st.button("Run fixture planner"):
        plan = MockPlanner().plan(query)
        st.caption("DEMO DATA · Deterministic MockPlanner output; not an LLM evaluation.")
        st.json(plan)
        st.write("Plan schema valid:", validate_plan(plan)["valid"])

    st.subheader("Change comparison preview")
    st.caption("DEMO DATA · Placeholder visualization uses fixture metrics only.")
    st.bar_chart({"T1 fixture": [FIXTURE_RESULTS["valid_pixel_t1_pct"]],
                  "T2 fixture": [FIXTURE_RESULTS["valid_pixel_t2_pct"]]})


if __name__ == "__main__":
    main()
