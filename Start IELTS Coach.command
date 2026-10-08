#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -x .venv/bin/python ]; then
  coach_python=".venv/bin/python"
elif [ -x ../../work/venv/bin/python ] && ../../work/venv/bin/python -c 'import streamlit, openai, pydantic' >/dev/null 2>&1; then
  coach_python="../../work/venv/bin/python"
else
  coach_base=""
  for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' >/dev/null 2>&1; then
      coach_base="$candidate"
      break
    fi
  done
  if [ -z "$coach_base" ]; then
    echo "Please install Python 3.10 or newer from python.org, then open this launcher again."
    read -r -p "Press Return to close."
    exit 1
  fi
  "$coach_base" -m venv .venv
  coach_python=".venv/bin/python"
fi

if ! "$coach_python" -c 'import streamlit, openai, pydantic' >/dev/null 2>&1; then
  echo "Installing the app's Python packages. This first setup needs internet."
  "$coach_python" -m pip install -r requirements.txt
fi

echo "Starting IELTS Coach. Keep this window open; press Control+C to stop."
exec "$coach_python" -m streamlit run app.py --server.address 127.0.0.1
