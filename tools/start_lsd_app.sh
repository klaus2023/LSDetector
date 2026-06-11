#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${LSD_APP_CONDA_ENV:-lsdetector}"
PORT="${LSD_APP_PORT:-8501}"
ADDRESS="${LSD_APP_ADDRESS:-0.0.0.0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${LSD_APP_ENV_FILE:-${APP_ROOT}/environment.yml}"

find_conda() {
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi
  for candidate in "$HOME/anaconda3/bin/conda" "$HOME/miniconda3/bin/conda" "/opt/conda/bin/conda"; do
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

CONDA_BIN="$(find_conda || true)"
if [ -z "${CONDA_BIN}" ]; then
  echo "ERROR: conda was not found. Install Anaconda or Miniconda, then rerun this script." >&2
  exit 1
fi

env_exists() {
  "$CONDA_BIN" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"
}

if env_exists; then
  echo "[LSDetector] Reusing conda environment: ${ENV_NAME}"
else
  echo "[LSDetector] Creating conda environment: ${ENV_NAME}"
  echo "[LSDetector] Environment file: ${ENV_FILE}"
  "$CONDA_BIN" env create -n "$ENV_NAME" -f "$ENV_FILE"
fi


CONDA_ENV_PREFIX="$($CONDA_BIN run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')"
export LD_LIBRARY_PATH="${CONDA_ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "[LSDetector] Checking core dependencies in ${ENV_NAME}"
"$CONDA_BIN" run -n "$ENV_NAME" env LSD_APP_CONDA_ENV="$ENV_NAME" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" python - <<'PY'
import importlib
import os
missing = []
for module in ("streamlit", "numpy", "rasterio", "geopandas", "cv2", "pywt"):
    try:
        importlib.import_module(module)
    except Exception as exc:
        missing.append(f"{module}: {exc}")
if missing:
    env_name = os.environ.get("LSD_APP_CONDA_ENV", "lsdetector")
    raise SystemExit(f"Missing dependencies in {env_name}:\n" + "\n".join(missing))
PY

echo "[LSDetector] Starting Streamlit on ${ADDRESS}:${PORT}"
cd "$APP_ROOT"
exec "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" PYTHONPATH="$APP_ROOT${PYTHONPATH:+:${PYTHONPATH}}" \
  streamlit run "$APP_ROOT/lsd_app/web/app.py" \
  --server.address "$ADDRESS" \
  --server.port "$PORT" \
  --server.headless true \
  --theme.base light \
  --theme.primaryColor "#e5484d" \
  --theme.backgroundColor "#f5f7fb" \
  --theme.secondaryBackgroundColor "#ffffff" \
  --theme.textColor "#111827"
