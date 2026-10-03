# GEO-INTEL architecture

The diagram shows the implemented project flow and the distinction between the
deterministic geospatial pipeline and the fixture-only demo/agent path. Hosted
raster execution and production integration are not verified.

```mermaid
flowchart LR
    subgraph Data[Hosted geospatial workflow]
        STAC[Sentinel-2 STAC provider] --> Fetch[Scene discovery and signing]
        Fetch --> Mask[Cloud and shadow mask]
        Mask --> Composite[T1/T2 median composites]
        DEM[Copernicus DEM] --> Terrain[Elevation, slope, aspect features]
        Composite --> Classify[5-class mapping]
        Terrain --> Classify
        Classify --> Post[Majority filter and MMU]
        Post --> Change[Transition and NDVI-difference outputs]
        Change --> Assoc[1 km association and hotspot summaries]
        Composite --> COG[COGs, metadata JSON, manifest]
        Assoc --> Outputs[Phase 4–5 outputs]
        Post --> Outputs
    end

    subgraph Agent[Query and evidence workflow]
        Query[User query] --> Planner[MockPlanner or opt-in live planner]
        Planner --> Validate[Plan allowlist and safety checks]
        Validate --> Retrieve[Fixture RAG retrieval]
        Validate --> Results[Supplied result JSON]
        Retrieve --> Answer[Constrained response with citations]
        Results --> Provenance[Number provenance and causal linter]
        Answer --> Provenance
    end

    subgraph Demo[Phase 7 demonstration]
        UI[Streamlit app] --> Fixtures[Synthetic fixture values]
        UI --> Planner
    end
```

The Streamlit app uses only synthetic fixture values and the deterministic mock
planner. It does not read the geospatial outputs shown in the hosted-workflow
subgraph. See [Phase 7 report](phase7_report.md) and [limitations](limitations.md).
