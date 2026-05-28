#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/job_week27_xmoe_config.sbatch"
OUT_DIR="/home/pham156/MoE/FuseMoE_poly/out/Week_27"
MANIFEST="${OUT_DIR}/week27_xmoe_grid_manifest.tsv"

LANES="${LANES:-8}"
USE_DEPENDENCIES="${USE_DEPENDENCIES:-0}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a SEEDS <<< "${SEEDS:-42 52 62 72}"
GATES=(softmax laplace poly)
XMOE_FLAGS=(False True)
SHARED_FLAGS=(0 1)
ROUTERS=(joint permod)
MODEL_SPECS=(
  "TS_CXR_Text 3 tstxtcxr"
  "TS_CXR_Text_ECG 4 tstxtcxrecg"
)

mkdir -p "$OUT_DIR"

echo -e "job_id\tlane\tconfig_id\tseed\tgate\txmoe\tshared\trouter\tmodeltype\tnum_modalities" > "$MANIFEST"

declare -a LANE_LAST_JOB
if [[ "$USE_DEPENDENCIES" == "1" ]]; then
  for ((i=0; i<LANES; i++)); do
    LANE_LAST_JOB[$i]=""
  done
fi

job_index=0
for seed in "${SEEDS[@]}"; do
  for gate in "${GATES[@]}"; do
    for xmoe in "${XMOE_FLAGS[@]}"; do
      for shared in "${SHARED_FLAGS[@]}"; do
        for router in "${ROUTERS[@]}"; do
          for model_spec in "${MODEL_SPECS[@]}"; do
            read -r modeltype num_modalities model_tag <<< "$model_spec"

            xmoe_tag="base"
            if [[ "$xmoe" == "True" ]]; then
              xmoe_tag="xmoe"
            fi
            config_id="s${seed}_${gate}_${xmoe_tag}_sh${shared}_${router}_${model_tag}"
            lane=$((job_index % LANES))

            output_path="${OUT_DIR}/%j_${config_id}.out"
            error_path="${OUT_DIR}/%j_${config_id}.err"
            if [[ "$USE_DEPENDENCIES" == "1" ]]; then
              job_name="w27_${lane}_${gate}_${xmoe_tag}"
            else
              job_name="w27_${gate}_${xmoe_tag}"
            fi

            export_args="ALL,CONFIG_ID=${config_id},SEED=${seed},GATING_FUNCTION=${gate},POLY_POWER=4,USE_XMOE_ROUTER=${xmoe},SHARED_EXPERTS=${shared},ROUTER=${router},MODELTYPE=${modeltype},NUM_MODALITIES=${num_modalities},TOTAL_EPOCHS=16,NUM_EXPERTS=4,TOP_K=2,BALANCE_COEF=0.01,ROUTER_PRINT_MODE=concise"

            sbatch_cmd=(sbatch --parsable --job-name "$job_name" --output "$output_path" --error "$error_path" --export "$export_args")
            if [[ "$USE_DEPENDENCIES" == "1" && -n "${LANE_LAST_JOB[$lane]}" ]]; then
              sbatch_cmd+=(--dependency "afterany:${LANE_LAST_JOB[$lane]}")
            fi
            sbatch_cmd+=("$JOB_SCRIPT")

            if [[ "$DRY_RUN" == "1" ]]; then
              echo "DRY_RUN lane=${lane} config=${config_id}: ${sbatch_cmd[*]}"
              job_id="DRYRUN_${job_index}"
            else
              job_id="$("${sbatch_cmd[@]}")"
              if [[ "$USE_DEPENDENCIES" == "1" ]]; then
                LANE_LAST_JOB[$lane]="$job_id"
              fi
            fi

            echo -e "${job_id}\t${lane}\t${config_id}\t${seed}\t${gate}\t${xmoe}\t${shared}\t${router}\t${modeltype}\t${num_modalities}" | tee -a "$MANIFEST"
            job_index=$((job_index + 1))
          done
        done
      done
    done
  done
done

echo "Submitted/planned ${job_index} Week27 config jobs."
if [[ "$USE_DEPENDENCIES" == "1" ]]; then
  echo "Dependency lanes: ${LANES}"
else
  echo "Dependency lanes: disabled"
fi
echo "Manifest: ${MANIFEST}"
