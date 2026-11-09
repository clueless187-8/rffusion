#!/usr/bin/env bash
# PostToolUse formatter for Edit|Write|MultiEdit. Fail-soft: never blocks.
input="$(cat)"
f="$(printf '%s' "$input" | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))
except Exception:
    print("")' 2>/dev/null)"
case "$f" in
  *.py) ruff format "$f" >/dev/null 2>&1 || true ;;
esac
exit 0
