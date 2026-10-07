# Verification Harness & Toolchain

This document defines the automated verification pipeline and enforcement mechanisms for this repository.

---

## Detected Toolchain

The project utilizes Python 3.9+ with Streamlit and Pandas. The verification toolchain comprises:

| Purpose | Tool / Command | Role |
| :--- | :--- | :--- |
| **Syntax & Compilation** | `python3 -m compileall -q src/` | Ensures syntax validity and byte-compilation across all source files. |
| **Test Runner** | `python3 -m unittest discover -s tests -p "test_*.py"` | Executes automated unit and regression tests. |
| **Linter** | `ruff check src/` (or `flake8 src/`) | Enforces PEP8, code hygiene, and style constraints. |
| **Type Checker** | `mypy src/ --ignore-missing-imports` | Validates static typing contracts and type hints. |

---

## Verification Pipeline (`./verify.sh`)

The authoritative gate for code acceptance is [`./verify.sh`](verify.sh). The script runs the following stages in sequence:

1. **Syntax & Bytecode Validation:**
   Verifies that every `.py` file compiles cleanly without syntax or indentation errors.
2. **Test Suite Execution:**
   Runs test discovery against the `tests/` directory (or workspace tests if configured).
3. **Lint Checking:**
   Runs `ruff` or `flake8` if installed in the environment; reports clean compilation otherwise.
4. **Static Type Checking:**
   Runs `mypy` if installed in the environment to verify type safety contracts.

---

## Autonomous Repair Mandate

* **15-Iteration Rule:** Agents must run `./verify.sh` autonomously and loop up to **15 times** to fix any syntax errors, test regressions, type errors, or lint failures before escalating to the user.
* **No Premature Escalation:** If `./verify.sh` fails with an actionable traceback or linter failure, the agent must inspect the failure, correct the code, and re-execute `./verify.sh`.
* **Escalate Only After 15 Exhausted Attempts:** Only if the verification loop fails continuously across 15 iterations may the agent halt and report the blocked state to the user.
