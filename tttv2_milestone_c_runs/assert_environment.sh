#!/usr/bin/env bash
# Re-assert the constants recorded in tttv2_milestone_c_evidence/perf/ENVIRONMENT.md.
# usage: assert_environment.sh [<label>]   -> writes a capture and diffs it against EXPECTED.txt
set -u
REPO=/proj_sw/user_dev/ctr-apbernal/tt-metal
cd "$REPO" || exit 91
LABEL="${1:-adhoc-$(date -u +%Y%m%dT%H%M%SZ)}"
EXP="$REPO/tttv2_milestone_c_evidence/perf/recon/_env0/EXPECTED.txt"
OUT="$REPO/tttv2_milestone_c_evidence/perf/recon/_envchecks/$LABEL"
mkdir -p "$OUT"
CUR="$OUT/CAPTURED.txt"

capture() {
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "tracked_dirty=$(git status --porcelain --untracked-files=no | wc -l)"
  echo "hostname=$(hostname)"
  echo "kernel=$(uname -r)"
  echo "boards_sysclass=$(ls /sys/class/tenstorrent | wc -l)"
  echo "tt_smi_version=$(tt-smi --version 2>&1 | head -1)"
  echo "hf_home=${HF_HOME:-<unset>}"
  echo "mesh_device=${MESH_DEVICE:-<unset>}"
  echo "llama_dir=${LLAMA_DIR:-<unset>}"
  echo "tt_cache_path=${TT_CACHE_PATH:-<unset>}"
  echo "tttv2_galaxy_ccl_trace=${TTTV2_GALAXY_CCL_TRACE:-<unset>}"
  echo "line_rs=${LINE_RS:-<unset>}"
  echo "line_ag=${LINE_AG:-<unset>}"
  echo "ckpt_llama=$(cat /localdev/ctr-apbernal/hf_data/hub/models--meta-llama--Llama-3.3-70B-Instruct/refs/main 2>&1)"
  echo "ckpt_qwen=$(cat /localdev/ctr-apbernal/hf_data/hub/models--Qwen--Qwen3-32B/refs/main 2>&1)"
  tt-smi -s > "$OUT/tt-smi-snapshot.json" 2>&1
  python - "$OUT/tt-smi-snapshot.json" <<'PY'
import json,sys,collections
d=json.load(open(sys.argv[1]))
h=d["host_info"]; s=d["host_sw_vers"]; devs=d["device_info"]
print(f"driver={h['Driver']}")
print(f"distro={h['Distro']}")
print(f"pyluwen={s['pyluwen']}")
print(f"tt_umd={s['tt_umd']}")
print(f"num_devices={len(devs)}")
fw=collections.Counter(json.dumps(dev["firmwares"],sort_keys=True) for dev in devs)
for blob,n in sorted(fw.items()):
    print(f"firmwares[x{n}]={blob}")
PY
}

capture > "$CUR" 2>&1

if [ ! -f "$EXP" ]; then
  echo "no EXPECTED.txt at $EXP; freezing this capture as the expectation"
  cp "$CUR" "$EXP"
  echo "FROZEN"
  exit 0
fi

if diff -u "$EXP" "$CUR" > "$OUT/DIFF.txt" 2>&1; then
  echo "ENVIRONMENT OK ($LABEL)"
  exit 0
else
  echo "ENVIRONMENT DRIFT ($LABEL) - see $OUT/DIFF.txt"
  cat "$OUT/DIFF.txt"
  exit 1
fi
