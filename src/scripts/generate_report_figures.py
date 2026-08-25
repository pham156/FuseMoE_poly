from __future__ import annotations

from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REPO_ROOT = Path("/home/pham156/MoE/FuseMoE_poly")
OUT_DIR = REPO_ROOT / "docs" / "figures"


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _setup() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["figure.dpi"] = 180
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8


def _normalize(df: pd.DataFrame, columns: list[str], invert: set[str] | None = None) -> pd.DataFrame:
    invert = invert or set()
    out = df.copy()
    for col in columns:
        vals = out[col].astype(float)
        lo, hi = vals.min(), vals.max()
        if hi - lo < 1e-12:
            out[col] = 0.5
            continue
        norm = (vals - lo) / (hi - lo)
        if col in invert:
            norm = 1.0 - norm
        out[col] = norm
    return out


def _radar_plot(
    df: pd.DataFrame,
    config_col: str,
    attribute_cols: list[str],
    attribute_labels: list[str],
    title: str,
    output_name: str,
    palette: list[str],
) -> None:
    angles = [n / float(len(attribute_cols)) * 2 * pi for n in range(len(attribute_cols))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.4, 5.4), subplot_kw={"polar": True}, constrained_layout=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(attribute_labels)
    ax.grid(alpha=0.5)

    for color, (_, row) in zip(palette, df.iterrows()):
        values = [float(row[c]) for c in attribute_cols]
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2.0, label=row[config_col])
        ax.fill(angles, values, color=color, alpha=0.10)

    ax.set_title(title, pad=20)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=True)
    fig.savefig(OUT_DIR / output_name, bbox_inches="tight")
    plt.close(fig)


def plot_routing_radar() -> None:
    df = _load_csv(
        REPO_ROOT / "out/Week_28/router_init_diagnostics/week28_permod_variant_grid_aggregate.csv"
    )
    keep = [
        ("Softmax, random, constant noise", "softmax", "random_constant_noise"),
        ("Laplace, random, constant noise", "laplace", "random_constant_noise"),
        ("Polynomial, random, constant noise", "poly", "random_constant_noise"),
        ("Softmax, zero, no noise", "softmax", "zero_no_noise"),
    ]
    rows = []
    for label, gate, variant in keep:
        sub = df[(df["gating_function"] == gate) & (df["variant"] == variant)]
        rows.append(
            {
                "Configuration": label,
                "Entropy": sub["test_gate_entropy_mean"].mean(),
                "Low top-1 weight": 1.0 - sub["test_gate_top1_weight_mean"].mean(),
                "Low margin": 1.0 - sub["test_gate_top12_margin_mean"].mean(),
                "Low mass deviation": 1.0 - sub["test_permod_mass_mean_maxdev_mean"].mean(),
                "Few dead experts": 1.0 - sub["test_permod_dead_expert_slots_lt001_mean"].mean() / 18.0,
                "Low similarity": 1.0 - sub["test_expert_offdiag_mean_mean"].mean(),
            }
        )
    radar = pd.DataFrame(rows)
    cols = [
        "Entropy",
        "Low top-1 weight",
        "Low margin",
        "Low mass deviation",
        "Few dead experts",
        "Low similarity",
    ]
    radar = _normalize(radar, cols)
    _radar_plot(
        radar,
        "Configuration",
        cols,
        ["Entropy", "Low top-1", "Low margin", "Low mass dev.", "Few dead", "Low sim."],
        "Routing profile by configuration",
        "routing_geometry_vs_utility.png",
        ["#1f77b4", "#2ca02c", "#d62728", "#4c4c4c"],
    )


def plot_semantic_radar() -> None:
    post = _load_csv(REPO_ROOT / "out/Week_30/analysis/week30_postmortem_aggregate.csv")
    post = post[post["task"].isin(["pheno-all-cxr-notes-ecg", "ihm-48-cxr-notes-ecg", "los-48-cxr-notes-ecg"])]
    base = post[post["family_name"] == "baseline"].set_index("task")
    sem = post[post["family_name"] == "D"].set_index("task")

    rows = [
        {
            "Configuration": "Baseline",
            "PHENO metric": float(base.loc["pheno-all-cxr-notes-ecg", "mean_test_primary"]),
            "IHM metric": float(base.loc["ihm-48-cxr-notes-ecg", "mean_test_primary"]),
            "LOS metric": float(base.loc["los-48-cxr-notes-ecg", "mean_test_primary"]),
            "PHENO semantic overlap": float(base.loc["pheno-all-cxr-notes-ecg", "mean_semantic_overlap"]),
            "IHM semantic overlap": float(base.loc["ihm-48-cxr-notes-ecg", "mean_semantic_overlap"]),
            "LOS semantic overlap": float(base.loc["los-48-cxr-notes-ecg", "mean_semantic_overlap"]),
        },
        {
            "Configuration": "Semantic guidance",
            "PHENO metric": float(sem.loc["pheno-all-cxr-notes-ecg", "mean_test_primary"]),
            "IHM metric": float(sem.loc["ihm-48-cxr-notes-ecg", "mean_test_primary"]),
            "LOS metric": float(sem.loc["los-48-cxr-notes-ecg", "mean_test_primary"]),
            "PHENO semantic overlap": float(sem.loc["pheno-all-cxr-notes-ecg", "mean_semantic_overlap"]),
            "IHM semantic overlap": float(sem.loc["ihm-48-cxr-notes-ecg", "mean_semantic_overlap"]),
            "LOS semantic overlap": float(sem.loc["los-48-cxr-notes-ecg", "mean_semantic_overlap"]),
        },
    ]
    radar = pd.DataFrame(rows)
    cols = [
        "PHENO metric",
        "IHM metric",
        "LOS metric",
        "PHENO semantic overlap",
        "IHM semantic overlap",
        "LOS semantic overlap",
    ]
    radar = _normalize(radar, cols)
    _radar_plot(
        radar,
        "Configuration",
        cols,
        ["PHENO", "IHM", "LOS", "PHENO overlap", "IHM overlap", "LOS overlap"],
        "Semantic guidance task profile",
        "semantic_initialization_pheno.png",
        ["#4c4c4c", "#2ca02c"],
    )


def plot_missingness_radar() -> None:
    df = _load_csv(REPO_ROOT / "out/Week_29/analysis/week29_causal_eval_mask_deltas_aggregate.csv")
    rows = []
    families = [
        ("Sparse MoE reference", "moe_ref"),
        ("Shared reference", "shared_ref"),
        ("Reconstruction reference", "recon_ref"),
    ]
    for label, family in families:
        lookup = df[df["family"] == family].set_index(["task", "mask"])
        rows.append(
            {
                "Configuration": label,
                "IHM no text+CXR": -float(lookup.loc[("ihm", "text_cxr"), "delta_metric_mean"]),
                "LOS no ECG": -float(lookup.loc[("los", "ecg"), "delta_metric_mean"]),
                "PHENO no ECG": -float(lookup.loc[("pheno", "ecg"), "delta_metric_mean"]),
                "PHENO no text+CXR": -float(lookup.loc[("pheno", "text_cxr"), "delta_metric_mean"]),
                "LOS no ECG AUC": -float(lookup.loc[("los", "ecg"), "delta_auc_mean"]),
                "PHENO no ECG AUC": float(lookup.loc[("pheno", "ecg"), "delta_auc_mean"]),
            }
        )
    radar = pd.DataFrame(rows)
    cols = [
        "IHM no text+CXR",
        "LOS no ECG",
        "PHENO no ECG",
        "PHENO no text+CXR",
        "LOS no ECG AUC",
        "PHENO no ECG AUC",
    ]
    radar = _normalize(radar, cols)
    _radar_plot(
        radar,
        "Configuration",
        cols,
        ["IHM no text+CXR", "LOS no ECG", "PHENO no ECG", "PHENO no text+CXR", "LOS ECG AUC", "PHENO ECG AUC"],
        "Missing-modality robustness profile",
        "missing_modality_robustness.png",
        ["#1f77b4", "#7f7f7f", "#ff7f0e"],
    )


def plot_family_radar() -> None:
    post = _load_csv(REPO_ROOT / "out/Week_30/analysis/week30_postmortem_aggregate.csv")
    pheno = post[post["task"] == "pheno-all-cxr-notes-ecg"].copy()
    pheno = pheno[pheno["family_name"].isin(["baseline", "A", "C", "D"])]
    pheno["Configuration"] = pheno["family_name"].map(
        {
            "baseline": "Baseline",
            "A": "Task-conditioned",
            "C": "Collaboration",
            "D": "Semantic guidance",
        }
    )
    radar = pd.DataFrame(
        {
            "Configuration": pheno["Configuration"],
            "Test metric": pheno["mean_test_primary"].astype(float),
            "Delta vs baseline": pheno["mean_delta_vs_baseline"].astype(float),
            "Gate entropy": pheno["mean_gate_entropy"].astype(float),
            "Low top-1 weight": 1.0 - pheno["mean_gate_top1_weight"].astype(float),
            "Low expert similarity": 1.0 - pheno["mean_expert_similarity"].astype(float),
            "Semantic overlap": pheno["mean_semantic_overlap"].astype(float),
        }
    )
    cols = [
        "Test metric",
        "Delta vs baseline",
        "Gate entropy",
        "Low top-1 weight",
        "Low expert similarity",
        "Semantic overlap",
    ]
    radar = _normalize(radar, cols)
    _radar_plot(
        radar,
        "Configuration",
        cols,
        ["Test metric", "Delta", "Entropy", "Low top-1", "Low similarity", "Sem. overlap"],
        "Extension-family profile on PHENO",
        "week30_family_summary.png",
        ["#4c4c4c", "#d62728", "#8c564b", "#2ca02c"],
    )


def plot_lingshu_delta() -> None:
    df = _load_csv(REPO_ROOT / "out/Week_27/lingshu_compare/lingshu_compare_results_108067.csv")
    rows = []
    base = df[df["arch"] == "base_shared"]
    ling = df[df["arch"] == "lingshu"]
    tasks = [
        ("ihm-48-cxr-notes-ecg", "In-hospital mortality"),
        ("los-48-cxr-notes-ecg", "Length of stay"),
        ("pheno-all-cxr-notes-ecg", "Phenotype prediction"),
    ]
    for task_id, task_label in tasks:
        b = base[base["task"] == task_id]
        l = ling[ling["task"] == task_id]
        if task_id.startswith("pheno"):
            base_best = b["best_macro_f1"].astype(float).mean()
            ling_best = l["best_macro_f1"].astype(float).mean()
            base_test = b["current_macro_f1"].astype(float).mean()
            ling_test = l["current_macro_f1"].astype(float).mean()
        else:
            base_best = b["best_f1"].astype(float).mean()
            ling_best = l["best_f1"].astype(float).mean()
            base_test = b["current_f1"].astype(float).mean()
            ling_test = l["current_f1"].astype(float).mean()
        rows.append({"Task": task_label, "Metric": "Best validation", "Delta": (ling_best - base_best) * 100.0})
        rows.append({"Task": task_label, "Metric": "Final test", "Delta": (ling_test - base_test) * 100.0})

    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.6, 3.9), constrained_layout=True)
    sns.barplot(
        data=plot_df,
        x="Delta",
        y="Task",
        hue="Metric",
        palette=["#1f77b4", "#ff7f0e"],
        ax=ax,
    )
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("Lingshu minus matched baseline (percentage points)")
    ax.set_ylabel("")
    ax.set_title("Lingshu comparison against matched baseline")
    ax.legend(title="")
    fig.savefig(OUT_DIR / "lingshu_delta_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _setup()
    plot_routing_radar()
    plot_semantic_radar()
    plot_missingness_radar()
    plot_family_radar()
    plot_lingshu_delta()


if __name__ == "__main__":
    main()
