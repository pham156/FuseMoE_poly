from pathlib import Path


REPO_HOME = Path("/home/pham156/MoE/FuseMoE_poly")
REPO_SCRATCH = Path("/scratch/gilbreth/pham156/MoE/FuseMoE_poly")

RESULTS_HOME_ROOT = REPO_HOME / "out" / "Week_31"
RESULTS_SCRATCH_ROOT = REPO_SCRATCH / "out" / "Week_31"
MANIFEST_ROOT = RESULTS_HOME_ROOT / "manifests"
REPORT_ROOT = RESULTS_HOME_ROOT / "reports"
AGGREGATE_ROOT = RESULTS_HOME_ROOT / "aggregates"


def ensure_week31_dirs() -> None:
    for path in [
        RESULTS_HOME_ROOT,
        RESULTS_SCRATCH_ROOT,
        MANIFEST_ROOT,
        REPORT_ROOT,
        AGGREGATE_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def config_output_dir(architecture: str, router_family: str, init_strategy: str, task: str, seed: int) -> Path:
    return (
        RESULTS_SCRATCH_ROOT
        / architecture
        / router_family
        / init_strategy
        / task
        / f"seed_{seed}"
    )
