#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_ROOT"

exec bash tools/start_lsd_app.sh
