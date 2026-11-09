#!/usr/bin/env bash
# PreToolUse guard for Bash. Reads the tool input JSON on stdin and blocks
# destructive / history-rewriting commands. Exit 2 = block with message.
set -u
input="$(cat)"
cmd="$(printf '%s' "$input" | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin).get("tool_input",{}).get("command",""))
except Exception:
    print("")' 2>/dev/null)"

block() { echo "guard.sh: blocked — $1" >&2; exit 2; }

case "$cmd" in
  *"--no-verify"*)                      block "--no-verify bypasses hooks" ;;
esac
if printf '%s' "$cmd" | grep -Eq 'git[[:space:]]+push[[:space:]]+.*(--force|-f([[:space:]]|$)|--force-with-lease)'; then
  block "force push"
fi
if printf '%s' "$cmd" | grep -Eq 'rm[[:space:]]+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)[[:space:]]+/([[:space:]]|$|\*)'; then
  block "rm -rf /"
fi
if printf '%s' "$cmd" | grep -Eq 'git[[:space:]]+reset[[:space:]]+--hard[[:space:]]+origin'; then
  block "git reset --hard origin"
fi
exit 0
