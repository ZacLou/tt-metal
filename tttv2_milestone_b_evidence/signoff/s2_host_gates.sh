#!/bin/bash
# mb-signoff attempt 2 host gates. No device is required or wanted by any of
# these; the two device-suite-including selections are run only because the
# brief names them literally, and every one of their errors is a cluster open
# against a mesh an operator has to fix.
set -u
cd /proj_sw/user_dev/ctr-apbernal/tt-metal
export HF_HOME=/localdev/ctr-apbernal/hf_data
S=tttv2_milestone_b_evidence/signoff/logs2
stamp() { { echo "=== $1 ==="; date -u; echo "commit: $(git rev-parse HEAD)"; echo "cmd: $2"; echo; } > "$3"; }

# --- gate 9: existing 1D model contract and demo-contract host tests ---
L=$S/s2_03_1d_contract_gate.log
FILES=$(ls models/common/tests/models/*/test_demo_contract.py models/common/tests/models/*/test_hf_adaptor.py 2>/dev/null | grep -v galaxy)
stamp "gate 9: 1D model contract + demo-contract host tests" "python -m pytest -q -rA <$(echo "$FILES" | wc -l) files>" "$L"
echo "$FILES" >> "$L"; echo >> "$L"
timeout --signal=TERM --kill-after=60 1800 python -m pytest -q -rA --color=no -p no:cacheprovider $FILES >> "$L" 2>&1
echo "exit=$?" >> "$L"

# --- the brief's regression command, device suites filtered out ---
L=$S/s2_04_host_regression_filtered.log
stamp "host regression, three directories, device suites filtered" \
  "python -m pytest -q models/common/tests/{modules,models,llm_runtime} --ignore-glob=*_wh_galaxy*.py --ignore=models/common/tests/models/galaxy/test_plans.py" "$L"
timeout --signal=TERM --kill-after=120 3000 python -m pytest -q -rf --color=no -p no:cacheprovider \
  models/common/tests/modules models/common/tests/models models/common/tests/llm_runtime \
  --ignore-glob="*_wh_galaxy*.py" --ignore=models/common/tests/models/galaxy/test_plans.py >> "$L" 2>&1
echo "exit=$?" >> "$L"

# --- the brief's regression command, literally, unfiltered ---
L=$S/s2_05_host_regression_literal.log
stamp "the brief's literal command, unfiltered" \
  "python -m pytest -q models/common/tests/modules models/common/tests/models models/common/tests/llm_runtime" "$L"
timeout --signal=TERM --kill-after=180 5400 python -m pytest -q -rf --color=no -p no:cacheprovider \
  models/common/tests/modules models/common/tests/models models/common/tests/llm_runtime >> "$L" 2>&1
echo "exit=$?" >> "$L"

echo "ALL DONE $(date -u)" > $S/s2_host_gates.done
