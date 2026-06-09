#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
STAMP="$(date +%Y%m%d_%H%M%S)"
WORK_DIR="${WORK_DIR:-${ROOT_DIR}/outputs/tool_smoke_${STAMP}}"
REAL_CONFIG=""
RUN_REAL=0
RUN_PYTEST=0
SKIP_EVAL=0

usage() {
  cat <<EOF
Usage:
  tools/test_lsd_tool_smoke.sh [options]

Options:
  --work-dir PATH       Test output directory. Default: outputs/tool_smoke_<timestamp>
  --real-config PATH    Optional real LSDSAM config, for dry-run by default
  --run-real            Actually run the real LSDSAM config; may need torch/peft/rasterio/PyWavelets and weights
  --run-pytest          Also run the focused pytest suite after the smoke test
  --skip-eval           Skip optional evaluation on the mock dataset
  -h, --help            Show this help

Examples:
  tools/test_lsd_tool_smoke.sh
  tools/test_lsd_tool_smoke.sh --real-config configs/example_lsdsam.yaml
  tools/test_lsd_tool_smoke.sh --real-config /path/to/real_lsdsam.yaml --run-real
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-dir)
      WORK_DIR="$2"
      shift 2
      ;;
    --real-config)
      REAL_CONFIG="$2"
      shift 2
      ;;
    --run-real)
      RUN_REAL=1
      shift
      ;;
    --run-pytest)
      RUN_PYTEST=1
      shift
      ;;
    --skip-eval)
      SKIP_EVAL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:${PYTHONPATH}}"

log() {
  printf '\n\033[1;34m>>> %s\033[0m\n' "$*"
}

ok() {
  printf '\033[1;32mOK\033[0m %s\n' "$*"
}

fail() {
  printf '\033[1;31mFAIL\033[0m %s\n' "$*" >&2
  exit 1
}

run() {
  log "$*"
  "$@"
}

tee_run() {
  local log_path="$1"
  shift
  log "$* | tee ${log_path}"
  "$@" | tee "$log_path"
}

check_file() {
  [[ -e "$1" ]] || fail "missing file: $1"
  ok "found $1"
}

check_no_file() {
  [[ ! -e "$1" ]] || fail "unexpected file exists: $1"
  ok "not created automatically: $1"
}

log "Checking Python dependencies used by the smoke dataset"
"$PYTHON_BIN" - <<'PY'
import importlib.util
missing = [name for name in ("numpy", "rasterio") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "Missing Python packages for the smoke test: "
        + ", ".join(missing)
        + "\nInstall the geospatial extras or run in the project's prepared environment."
    )
PY
ok "numpy/rasterio available"

mkdir -p "$WORK_DIR"
DATA_DIR="$WORK_DIR/mock_dataset"
MOCK_CONFIG="$WORK_DIR/config.mock.json"
MOCK_NORMALIZED="$WORK_DIR/normalized.mock.json"
MOCK_OUTPUT="$WORK_DIR/deployment"
REGISTRY_ROOT="$WORK_DIR"
MODELS_JSON="$WORK_DIR/models.json"
DATASETS_JSON="$WORK_DIR/datasets.json"

run "$PYTHON_BIN" -m examples.generate_mock_dataset \
  --output "$DATA_DIR" \
  --size 64 \
  --seed 42 \
  --with-truth \
  --config-format json \
  --overwrite

log "Writing smoke config with explicit bgr_first3 channel order"
"$PYTHON_BIN" - "$DATA_DIR/config.json" "$MOCK_CONFIG" "$MOCK_OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

source_config, target_config, output_dir = map(Path, sys.argv[1:4])
config = json.loads(source_config.read_text(encoding="utf-8"))
config.setdefault("input", {})["channel_order"] = "bgr_first3"
config.setdefault("model", {})["name"] = "mock"
config.setdefault("model", {})["checkpoint_path"] = None
config.setdefault("output", {})["dir"] = str(output_dir)
config.setdefault("output", {})["vector_formats"] = ["gpkg", "geojson"]
Path(target_config).write_text(json.dumps(config, indent=2), encoding="utf-8")
print(target_config)
PY

tee_run "$MODELS_JSON" "$PYTHON_BIN" -m lsd_app.cli.main models list --project-root "$ROOT_DIR"
"$PYTHON_BIN" - "$MODELS_JSON" <<'PY'
import json
import sys
rows = json.loads(open(sys.argv[1], encoding="utf-8").read())
by_id = {row["model_id"]: row for row in rows}
assert by_id["lsdsam"]["input_mode"] == "rgb_wavelets"
assert by_id["lsdsam"]["edge_mode"] == "wavelets"
assert by_id["mock"]["input_mode"] == "rgb"
print("validated model repository: LSDSAM=RGB+wavelets, mock=RGB")
PY

tee_run "$MOCK_NORMALIZED" "$PYTHON_BIN" -m lsd_app.cli.main config --config "$MOCK_CONFIG" --no-create-dirs
"$PYTHON_BIN" - "$MOCK_NORMALIZED" <<'PY'
import json
import sys
config = json.loads(open(sys.argv[1], encoding="utf-8").read())
assert config["channel_order"] == "bgr_first3"
assert config["input_mode"] == "rgb"
assert config["truth_mask_path"]
print("validated mock config: channel_order=bgr_first3, input_mode=rgb")
PY

run "$PYTHON_BIN" -m lsd_app.cli.main datasets add \
  --registry-root "$REGISTRY_ROOT" \
  --name "Mock DAFT evaluation" \
  --class DAFT \
  --imagery "$DATA_DIR/source.tif" \
  --label "$DATA_DIR/truth.tif" \
  --has-gt \
  --channel-order bgr_first3 \
  --tile-schema "tile_size=64;stride=64" \
  --source "smoke-test" \
  --description "Smoke-test DAFT dataset with local ground truth"

run "$PYTHON_BIN" -m lsd_app.cli.main datasets add \
  --registry-root "$REGISTRY_ROOT" \
  --name "Mock TAFT seed" \
  --class TAFT \
  --imagery "$DATA_DIR/source.tif" \
  --label "$DATA_DIR/truth.tif" \
  --has-gt \
  --channel-order bgr_first3 \
  --tile-schema "tile_size=64;stride=64" \
  --source "smoke-test" \
  --description "Smoke-test pretraining repository seed"

tee_run "$DATASETS_JSON" "$PYTHON_BIN" -m lsd_app.cli.main datasets list --registry-root "$REGISTRY_ROOT"
"$PYTHON_BIN" - "$DATASETS_JSON" <<'PY'
import json
import sys
rows = json.loads(open(sys.argv[1], encoding="utf-8").read())
classes = {row["dataset_class"] for row in rows}
assert "DAFT" in classes
assert "TAFT" in classes
assert all(row["has_gt"] for row in rows)
print("validated dataset registry: DAFT/TAFT entries both have ground truth")
PY

tee_run "$WORK_DIR/run.mock.log" "$PYTHON_BIN" -m lsd_app.cli.main run --config "$MOCK_CONFIG" --model mock

check_file "$MOCK_OUTPUT/patch_index.geojson"
check_file "$MOCK_OUTPUT/infer_manifest.json"
check_file "$MOCK_OUTPUT/rasters/probability.tif"
check_file "$MOCK_OUTPUT/rasters/mask.tif"
check_file "$MOCK_OUTPUT/vectors/landslides.gpkg"
check_file "$MOCK_OUTPUT/vectors/landslides.geojson"
check_file "$MOCK_OUTPUT/metrics/vector_stats.json"
check_file "$MOCK_OUTPUT/visualizations/overlay.png"
check_file "$REGISTRY_ROOT/registry/prediction_runs.json"
check_no_file "$MOCK_OUTPUT/metrics/evaluation.json"

"$PYTHON_BIN" - "$MOCK_OUTPUT/vectors/landslides.gpkg" <<'PY'
import sqlite3
import sys
with sqlite3.connect(sys.argv[1]) as conn:
    count = conn.execute("SELECT COUNT(*) FROM landslides").fetchone()[0]
print(f"validated vector output: polygons={count}")
PY

if [[ "$SKIP_EVAL" -eq 0 ]]; then
  tee_run "$WORK_DIR/evaluate.mock.log" "$PYTHON_BIN" -m lsd_app.cli.main evaluate --config "$MOCK_CONFIG"
  check_file "$MOCK_OUTPUT/metrics/evaluation.json"
  check_file "$MOCK_OUTPUT/metrics/evaluation.csv"
  "$PYTHON_BIN" - "$MOCK_OUTPUT/metrics/evaluation.json" <<'PY'
import json
import sys
metrics = json.loads(open(sys.argv[1], encoding="utf-8").read())
for key in ("iou", "precision", "recall", "f1", "confusion_matrix"):
    assert key in metrics
print(
    "validated optional evaluation: "
    f"iou={metrics['iou']}, f1={metrics['f1']}"
)
PY
fi

LSDSAM_CONFIG="$WORK_DIR/config.lsdsam.dry_run.json"
LSDSAM_NORMALIZED="$WORK_DIR/normalized.lsdsam.json"
log "Writing LSDSAM dry-run config to verify fixed wavelets logic"
"$PYTHON_BIN" - "$MOCK_CONFIG" "$LSDSAM_CONFIG" "$WORK_DIR/lsdsam_dry_output" "$ROOT_DIR" <<'PY'
import json
import sys
from pathlib import Path

base_config, target_config, output_dir, root_dir = map(Path, sys.argv[1:5])
config = json.loads(base_config.read_text(encoding="utf-8"))
config["model"] = {
    "name": "lsdsam",
    "checkpoint_path": str(
        root_dir / "LSDSAM_finetune/hunansat_Finetune_SAM/models_save/30/LoRA2_DA_2025.pth"
    ),
    "options": {"edge": "sobel_should_be_overridden"},
}
config["output"]["dir"] = str(output_dir)
Path(target_config).write_text(json.dumps(config, indent=2), encoding="utf-8")
print(target_config)
PY

tee_run "$LSDSAM_NORMALIZED" "$PYTHON_BIN" -m lsd_app.cli.main config --config "$LSDSAM_CONFIG" --no-create-dirs
"$PYTHON_BIN" - "$LSDSAM_NORMALIZED" <<'PY'
import json
import sys
config = json.loads(open(sys.argv[1], encoding="utf-8").read())
assert config["input_mode"] == "rgb_wavelets"
assert config["adapter_options"]["edge"] == "wavelets"
print("validated LSDSAM config normalization: input_mode=rgb_wavelets, edge=wavelets")
PY
tee_run "$WORK_DIR/run.lsdsam.dry_run.log" "$PYTHON_BIN" -m lsd_app.cli.main run --config "$LSDSAM_CONFIG" --dry-run

if [[ -n "$REAL_CONFIG" ]]; then
  REAL_CONFIG_ABS="$(cd "$(dirname "$REAL_CONFIG")" && pwd)/$(basename "$REAL_CONFIG")"
  [[ -e "$REAL_CONFIG_ABS" ]] || fail "real config not found: $REAL_CONFIG"
  REAL_OUTPUT="$WORK_DIR/real_lsdsam_output"
  tee_run "$WORK_DIR/run.real_lsdsam.dry_run.log" \
    "$PYTHON_BIN" -m lsd_app.cli.main run \
      --config "$REAL_CONFIG_ABS" \
      --model lsdsam \
      --output-dir "$REAL_OUTPUT" \
      --dry-run
  if [[ "$RUN_REAL" -eq 1 ]]; then
    tee_run "$WORK_DIR/run.real_lsdsam.log" \
      "$PYTHON_BIN" -m lsd_app.cli.main run \
        --config "$REAL_CONFIG_ABS" \
        --model lsdsam \
        --output-dir "$REAL_OUTPUT"
    check_file "$REAL_OUTPUT/vectors/landslides.gpkg"
    check_file "$REAL_OUTPUT/visualizations/overlay.png"
  else
    ok "real LSDSAM config checked in dry-run mode; pass --run-real to execute it"
  fi
fi

if [[ "$RUN_PYTEST" -eq 1 ]]; then
  run "$PYTHON_BIN" -m pytest -q \
    tests/test_config.py \
    tests/test_model_adapters.py \
    tests/test_registry.py \
    tests/test_cli_run.py \
    tests/test_web_app.py
fi

REPORT_HTML="$WORK_DIR/report.html"
cat > "$REPORT_HTML" <<EOF
<!doctype html>
<meta charset="utf-8">
<title>LSD_APP smoke-test report</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; line-height: 1.5; }
  code { background: #f3f4f6; padding: 2px 5px; border-radius: 4px; }
  img { max-width: 720px; width: 100%; border: 1px solid #ddd; }
</style>
<h1>LSD_APP smoke-test report</h1>
<p>Work dir: <code>${WORK_DIR}</code></p>
<p>Deployment output: <code>${MOCK_OUTPUT}</code></p>
<p>Prediction polygons: <code>${MOCK_OUTPUT}/vectors/landslides.gpkg</code></p>
<p>Overlay visualization:</p>
<img src="${MOCK_OUTPUT}/visualizations/overlay.png" alt="Landslide overlay">
EOF

cat <<EOF

Smoke test completed.

Key outputs:
  Work dir:              ${WORK_DIR}
  Mock config:           ${MOCK_CONFIG}
  Deployment polygons:   ${MOCK_OUTPUT}/vectors/landslides.gpkg
  Overlay PNG:           ${MOCK_OUTPUT}/visualizations/overlay.png
  Optional metrics:      ${MOCK_OUTPUT}/metrics/evaluation.json
  Dataset registry:      ${REGISTRY_ROOT}/registry/datasets.json
  HTML report:           ${REPORT_HTML}

Open the HTML report in a browser to inspect the overlay visualization.
EOF
