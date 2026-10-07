# Engineering Principles & Guidelines

These principles dictate operational standards, escalation triggers, and development defaults across the codebase.

---

## Escalation Triggers

Autonomous execution must immediately pause and prompt the user for confirmation when any of the following triggers occur:

1. **Adding External Dependencies:**
   - Any modification to `requirements.txt` or introduction of a third-party package not currently installed in the environment.
2. **Modifying Public API Routes or Function Contracts:**
   - Any modification to the signature, return type, or behavioral contract of core functions:
     - `src/analyzer.py`: `analyze_dataset(file)`
     - `src/llm_service.py`: `get_ai_insights(summary)`
     - `src/database.py`: `init_db()`, `save_entry(filename, insights)`, `get_all_entries()`, `delete_entry(entry_id)`
3. **Altering Database Migration & Storage Schemas:**
   - Any change to the Google Sheets columnar schema (`timestamp`, `filename`, `insights`), column ordering, or table definition.

---

## Code Defaults

All code written or modified in this repository must adhere to the following defaults:

### 1. Explicit Domain Errors over Silent Null Fallbacks
* Never return silent `None`, empty dictionaries `{}` or suppress exceptions when an operation fails.
* Surface clear, actionable error descriptions (or domain error types) so callers can diagnose the root cause immediately.
* Do not swallow network, parsing, or file read errors with catch-all handlers that pretend nothing happened.

### 2. Avoid Premature Abstractions for Single-Use Logic
* Do not introduce multi-layered class hierarchies, generic design patterns, factories, or wrapper classes for single-use functionality.
* Prefer clean, composable, and idiomatic Python functions over speculative over-engineering.
* Refactor only when duplicated patterns appear across 3 or more distinct call sites.

### 3. Zero Untyped Bypasses
* All new or modified functions must feature complete Python type annotations (parameters and return types).
* Avoid untyped fallbacks such as `Any`, unvalidated `**kwargs`, or `# type: ignore` directives without explicit justification.
* Maintain type-safety across data transformations and dictionary structures.
