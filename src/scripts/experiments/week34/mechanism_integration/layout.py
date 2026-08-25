from pathlib import Path


REPO_HOME = Path("/home/pham156/MoE/FuseMoE_poly")
REPO_SCRATCH = Path("/scratch/gilbreth/pham156/MoE/FuseMoE_poly")

WEEK34_HOME = REPO_HOME / "out" / "Week_34" / "mechanism_integration"
WEEK34_SCRATCH = REPO_SCRATCH / "out" / "Week_34" / "mechanism_integration"

MANIFEST_ROOT = WEEK34_HOME / "manifests"
REPORT_ROOT = WEEK34_HOME / "reports"
AGGREGATE_ROOT = WEEK34_HOME / "aggregates"
LOG_ROOT = WEEK34_HOME / "logs"


def ensure_week34_dirs() -> None:
    for path in (
        WEEK34_HOME,
        WEEK34_SCRATCH,
        MANIFEST_ROOT,
        REPORT_ROOT,
        AGGREGATE_ROOT,
        LOG_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def train_output_dir(
    family_name: str,
    variant_name: str,
    architecture: str,
    task_short: str,
    seed: int,
) -> Path:
    return WEEK34_SCRATCH / family_name / variant_name / architecture / task_short / f"seed_{seed}"
