#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, re, ast
from pathlib import Path
import numpy as np
import pandas as pd


# ---------- helpers ----------
def parse_gating(g):
    """['laplace','poly'] or messy strings -> 'laplace-poly'."""
    if isinstance(g, list):
        return "-".join(map(str, g))
    if pd.isna(g):
        return "N/A"
    s = str(g).replace('""','"').replace("“","\"").replace("”","\"").strip()
    # Try literal list first
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return "-".join(map(str, obj))
    except Exception:
        pass
    # Fallback: regex tokens in order of appearance
    toks = re.findall(r"\b(laplace|softmax|poly)\b", s.lower())
    return "-".join(toks) if toks else "N/A"


def pct3(series: pd.Series) -> pd.Series:
    """If values look like probabilities in [0,1], scale ×100; else just round."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().any():
        q1, q99 = s.quantile(0.01), s.quantile(0.99)
        if (pd.notna(q1) and pd.notna(q99) and 0.0 <= q1 <= 1.0 and 0.0 <= q99 <= 1.0):
            s = (s * 100.0).round(3)
        else:
            s = s.round(3)
    return s.where(s.notna(), series)


def norm_colkey(s: str) -> str:
    """Normalize column name for fuzzy matching."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def normalize_router_value(v) -> str:
    """Map messy router values to canonical labels."""
    if pd.isna(v):
        return "N/A"
    s = str(v).strip().lower()
    if re.search(r"\bjoint\b", s):
        return "joint"
    if re.search(r"\bper[-_\s]*mod\b|\bpermod\b|\bper[-_\s]*modal\b", s):
        return "permod"
    if re.search(r"\bdisjoint\b", s):
        return "disjoint"
    return s if s else "N/A"


def to_bool_like(v):
    """Coerce various truthy/falsey strings/numbers to bool or None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    # try numeric
    try:
        n = float(s)
        if n == 0:
            return False
        if n == 1:
            return True
    except Exception:
        pass
    return None


# ---------- main ----------
def main(inp, outp):
    in_path  = Path(inp).expanduser().resolve()
    out_path = Path(outp).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[clean_results] Reading  : {in_path}")
    print(f"[clean_results] Will save: {out_path}")

    df = pd.read_csv(in_path)
    df.columns = [c.strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}
    print(f"[clean_results] Columns  : {list(df.columns)}")

    # --- poly_power (required) ---
    if "poly_power" not in lower_map:
        raise SystemExit("ERROR: Couldn't find 'poly_power' column (case-insensitive).")
    poly_col = lower_map["poly_power"]
    df["poly_power"] = pd.to_numeric(df[poly_col], errors="coerce")

    # --- gating_function -> 'a-b' ---
    gf_col = lower_map.get("gating_function")
    if gf_col:
        df["gating_function"] = df[gf_col].apply(parse_gating)
    else:
        df["gating_function"] = "N/A"

    # --- router_type (fuzzy find + normalize) ---
    rt_col = lower_map.get("router_type")
    if rt_col is None:
        keymap = {norm_colkey(c): c for c in df.columns}
        for key in ("routertype", "router", "routerkind", "routermode"):
            if key in keymap:
                rt_col = keymap[key]
                break
        if rt_col is None:
            for c in df.columns:
                lc = c.lower()
                if "router" in lc and "type" in lc:
                    rt_col = c
                    break

    if rt_col:
        print(f"[clean_results] Using router column: {rt_col}")
        df["router_type"] = df[rt_col].apply(normalize_router_value)
    else:
        print("[clean_results] No router-type-like column found; setting to 'N/A'")
        df["router_type"] = "N/A"

    # --- noisy_gating -> noise ("noisy"/"not noisy"/"N/A") ---
    ng_col = lower_map.get("noisy_gating")
    if ng_col is None:
        # light fuzzy find if someone wrote noisy or noise_gating etc.
        keymap = {norm_colkey(c): c for c in df.columns}
        for key in ("noisygating", "noisegating", "noisy", "noise"):
            if key in keymap:
                ng_col = keymap[key]
                break

    if ng_col:
        print(f"[clean_results] Using noise column: {ng_col}")
        noise_bool = df[ng_col].apply(to_bool_like)
        df["noise"] = noise_bool.map({True: "noisy", False: "not noisy"}).fillna("N/A")
    else:
        print("[clean_results] No noisy_gating-like column found; setting noise to 'N/A'")
        df["noise"] = "N/A"

    # --- metrics auto-scale (test/*) ---
    metric_cols = [c for c in df.columns if c.startswith("test/")]
    for c in metric_cols:
        df[c] = pct3(df[c])

    # --- build output view ---
    out = df[["gating_function", "poly_power", "router_type", "noise"] + metric_cols].copy()

    # If gating_function does not include 'poly', set poly_power = NaN (display as N/A later)
    has_poly = out["gating_function"].str.contains("poly", case=False, na=False)
    out.loc[~has_poly, "poly_power"] = np.nan

    # --- sorting: router -> gating group -> poly power ---
    out["_power_num"] = pd.to_numeric(out["poly_power"], errors="coerce")
    out["_g"] = out["gating_function"].str.lower()
    out["_r"] = out["router_type"].str.lower()

    router_priority = {"joint": 0, "permod": 1}
    out["_r_rank"] = out["_r"].map(router_priority).fillna(100).astype(int)

    gating_priority = {"laplace-poly": 0, "softmax-poly": 1}
    out["_g_rank"] = out["_g"].map(gating_priority).fillna(100).astype(int)

    out = out.sort_values(
        by=["_r_rank", "_g_rank", "_power_num"],
        ascending=[True, True, True],
        kind="mergesort",
    )

    # Format poly_power for display
    out["poly_power"] = out["_power_num"].apply(lambda v: "N/A" if pd.isna(v) else f"{float(v):.3f}")

    # Cleanup helper cols
    out = out.drop(columns=["_r_rank", "_g_rank", "_r", "_g", "_power_num"])

    # Add 1-based index
    out.insert(0, "index", range(1, len(out) + 1))

    # Save
    out.to_csv(out_path, index=False)
    print(f"[clean_results] Wrote    : {out_path} (size: {out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_results.py <input.csv> <output.csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

