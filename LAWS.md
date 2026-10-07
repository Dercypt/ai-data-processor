# Laws of the System

## Primary Domain
**AI Data Processor** is an analytics application that ingests tabular datasets (CSV), performs automated data cleaning, computes statistical summaries, generates executive business insights via Google Gemini, and synchronizes analysis history to a Google Sheet cloud database.

---

## Non-Negotiable System Invariants

### Law 0: Governance & Law Immutability (Strict Meta-Law)
Files located in `tests/laws/` and `LAWS.md` are **strictly immutable and read-only to all agents**. 
* Under no circumstances may an agent alter, delete, comment out, xfail, skip, or weaken these laws or their verification tests.
* Any proposed alteration to `LAWS.md` or `tests/laws/` requires explicit human approval outside the agent execution loop.

### Law 1: Data Privacy & LLM Boundary Protection
**Raw tabular dataset rows and individual records must NEVER be sent to external LLM services.**
* Only aggregated statistical summaries (`columns`, `rows`, and `describe()` dictionary statistics) may be transmitted to `get_ai_insights()`.
* Raw user data must remain isolated within the local runtime environment.
* Protected code: [`src/llm_service.py`](src/llm_service.py), [`src/main.py`](src/main.py).

### Law 2: Data Cleaning Integrity
**Automated data cleaning must never drop rows or columns silently and must eliminate missing values.**
* `analyze_dataset()` must preserve the complete row count and original column structure of the uploaded CSV.
* Imputation routines must ensure numerical and categorical data structures remain intact without data loss.
* Protected code: [`src/analyzer.py`](src/analyzer.py).

### Law 3: Analyzer Contract & Fault Containment
**The data analyzer must always adhere to the uniform return signature `(df, summary, error)`.**
* Valid operations return `(df, summary, None)`.
* Invalid operations, corrupted files, or parsing failures must return `(None, None, str(e))`.
* Ingestion errors must never bubble up as uncaught exceptions that terminate or crash the Streamlit web process.
* Protected code: [`src/analyzer.py`](src/analyzer.py), [`src/main.py`](src/main.py).

### Law 4: Persistence Schema & Audit Trail Integrity
**The cloud persistence layer must strictly maintain the historical schema `[timestamp, filename, insights]`.**
* Timestamps must adhere to `%Y-%m-%d %H:%M:%S` format.
* Historical records must be append-only unless an explicit user-initiated deletion is triggered.
* A network failure or failed read from the database must **never** result in wiping or overwriting the existing sheet with an empty dataset.
* Protected code: [`src/database.py`](src/database.py).

### Law 5: Secret & Credential Boundary
**API keys and service account credentials must never be committed to source control or leaked to logs.**
* All secrets (`GOOGLE_API_KEY`, Google service account private keys) must strictly originate from `.streamlit/secrets.toml` or `.env`.
* Credentials must never appear in error messages, UI alerts, application logs, or git commits.
* Protected code: [`.gitignore`](.gitignore), [`src/llm_service.py`](src/llm_service.py), [`src/database.py`](src/database.py).
