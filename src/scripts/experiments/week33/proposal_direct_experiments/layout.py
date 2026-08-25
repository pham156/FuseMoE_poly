from pathlib import Path


REPO_HOME = Path("/home/pham156/MoE/FuseMoE_poly")
REPO_SCRATCH = Path("/scratch/gilbreth/pham156/MoE/FuseMoE_poly")

WEEK33_HOME = REPO_HOME / "out" / "Week_33" / "proposal_direct_experiments"
WEEK33_SCRATCH = REPO_SCRATCH / "out" / "Week_33" / "proposal_direct_experiments"
MANIFEST_ROOT = WEEK33_HOME / "manifests"
REPORT_ROOT = WEEK33_HOME / "reports"
AGGREGATE_ROOT = WEEK33_HOME / "aggregates"
LOG_ROOT = WEEK33_HOME / "logs"
TARGET_ROOT = WEEK33_HOME / "targets"
MASKED_EVAL_ROOT = WEEK33_HOME / "masked_eval"


def ensure_week33_dirs() -> None:
    for path in (
        WEEK33_HOME,
        WEEK33_SCRATCH,
        MANIFEST_ROOT,
        REPORT_ROOT,
        AGGREGATE_ROOT,
        LOG_ROOT,
        TARGET_ROOT,
        MASKED_EVAL_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def train_output_dir(experiment_group: str, variant_name: str, architecture: str, task_short: str, seed: int) -> Path:
    return WEEK33_SCRATCH / experiment_group / variant_name / architecture / task_short / f"seed_{seed}"


def masked_eval_output_dir(experiment_group: str, variant_name: str, architecture: str, task_short: str, seed: int, mask_name: str) -> Path:
    return MASKED_EVAL_ROOT / experiment_group / variant_name / architecture / task_short / f"seed_{seed}" / mask_name

