#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/config/demo_config.json"

if [[ -z "${DISPLAY:-}" ]]; then
  for candidate_display in :1 :0; do
    if [[ -S "/tmp/.X11-unix/X${candidate_display#:}" ]]; then
      export DISPLAY="${candidate_display}"
      break
    fi
  done
fi

if [[ -z "${XAUTHORITY:-}" ]]; then
  for candidate_auth in \
    "/run/user/$(id -u)/gdm/Xauthority" \
    "${HOME}/.Xauthority"
  do
    if [[ -f "${candidate_auth}" ]]; then
      export XAUTHORITY="${candidate_auth}"
      break
    fi
  done
fi

if [[ -d /usr/share/fonts ]] && [[ -z "${QT_QPA_FONTDIR:-}" ]]; then
    export QT_QPA_FONTDIR=/usr/share/fonts/truetype/dejavu:/usr/share/fonts/truetype
fi

echo "Launching demo with DISPLAY=${DISPLAY:-unset} XAUTHORITY=${XAUTHORITY:-unset}"
python3 -u "${SCRIPT_DIR}/app/behavior_demo.py" --config "${CONFIG_PATH}" "$@"
