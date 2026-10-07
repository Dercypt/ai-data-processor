#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo " Running Verification Harness"
echo "=========================================="

# 1. Syntax & Compilation
echo "[1/4] Checking Python syntax & compilation..."
python3 -m compileall -q src/
echo "✓ Syntax & compilation clean."

# 2. Test Suite Execution
echo "[2/4] Running test suite..."
if [ -d "tests" ]; then
    python3 -m unittest discover -s tests -p "test_*.py" -v
else
    echo "Notice: 'tests' directory not found. Discovering workspace tests..."
    python3 -m unittest discover -s . -p "test_*.py"
fi
echo "✓ Test execution passed."

# 3. Lint Checking
echo "[3/4] Lint checking..."
if command -v ruff >/dev/null 2>&1; then
    ruff check src/
    echo "✓ Ruff lint passed."
elif python3 -m ruff --version >/dev/null 2>&1; then
    python3 -m ruff check src/
    echo "✓ Ruff lint passed."
elif command -v flake8 >/dev/null 2>&1; then
    flake8 src/
    echo "✓ Flake8 lint passed."
elif python3 -m flake8 --version >/dev/null 2>&1; then
    python3 -m flake8 src/
    echo "✓ Flake8 lint passed."
else
    echo "Notice: No dedicated linter (ruff/flake8) installed in environment. Syntax compileall verified."
fi

# 4. Static Type Checking
echo "[4/4] Type checking..."
if command -v mypy >/dev/null 2>&1; then
    mypy src/ --ignore-missing-imports
    echo "✓ Mypy type check passed."
elif python3 -m mypy --version >/dev/null 2>&1; then
    python3 -m mypy src/ --ignore-missing-imports
    echo "✓ Mypy type check passed."
else
    echo "Notice: Mypy is not installed in environment. Skipping static type checking."
fi

echo "=========================================="
echo " Verification Pipeline Passed Successfully"
echo "=========================================="
