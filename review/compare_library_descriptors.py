"""Compare per-conformer descriptor tables from two DescriPyTor versions, and compare each,
after the notebook's Boltzmann aggregation, against a reference descriptor CSV.

Usage (repo root, gp_env):
    python review/compare_library_descriptors.py lib_vendored.csv lib_upstream.csv 20260423_LNiH2_..._descriptors.csv
"""
import sys, os, numpy as np, pandas as pd

vend_csv, up_csv, ref_csv = sys.argv[1:4]
num = pd.read_csv("atom_numbering_results.csv")[["path", "SCF_energy"]]

def load(p):
    df = pd.read_csv(p).merge(num, left_on="feather_dir", right_on="path", how="left")
    df["inchi_folder"] = df["feather_dir"].str.rstrip("/").str.replace(r"/conf_\d+$", "", regex=True)
    return df

def aggregate(df):
    """Verbatim re-implementation of the notebook's Boltzmann aggregation cell."""
    T, kB = 298.15, 3.166811563e-6
    cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ("SCF_energy", "n_loaded")]
    rows = []
    for inchi_folder, g in df.groupby("inchi_folder"):
        e = g["SCF_energy"]; idx_low = e.idxmin()
        w = np.exp(-((e - e.min()) / (kB * T))); w = w / w.sum()
        out = {"inchi": os.path.basename(inchi_folder)}
        for c in cols:
            v = g[c]; valid = v.notna()
            out[f"{c}_min"], out[f"{c}_max"], out[f"{c}_lowE"] = v.min(), v.max(), g.loc[idx_low, c]
            out[f"{c}_boltz"] = np.average(v[valid], weights=w[valid]) if valid.any() else np.nan
        rows.append(out)
    return pd.DataFrame(rows).set_index("inchi")

def report(a, b, label, tol=1e-6):
    common = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c])]
    a, b = a[common], b.reindex(a.index)[common]
    diff = (a - b).abs()
    changed = (diff > tol) | (a.isna() != b.isna())
    fam = lambda c: c.split("_")[0] if not c.startswith(("vib_freq_bend", "vib_amp_bend", "charge_NBO_diff", "loc_B5", "B1-B5")) else "_".join(c.split("_")[:3]) if c.startswith("charge_NBO_diff") else "_".join(c.split("_")[:2]) if c.startswith("loc_B5") else "_".join(c.split("_")[:3]) if c.startswith("vib_") else "B1-B5_angle"
    tab = pd.DataFrame({"family": [fam(c) for c in common], "n_changed": changed.sum().values,
                        "n_values": a.notna().sum().values, "max_abs_diff": diff.max().values}, index=common)
    summ = tab.groupby("family").agg(columns=("n_changed", "size"), values=("n_values", "sum"),
                                     values_changed=("n_changed", "sum"), max_abs_diff=("max_abs_diff", "max"))
    summ["pct_changed"] = (100 * summ["values_changed"] / summ["values"]).round(1)
    print(f"\n=== {label}: {len(common)} common columns, {len(a)} rows, tolerance {tol} ===")
    print(summ.sort_values("pct_changed", ascending=False).to_string())
    return tab

vend, up = load(vend_csv), load(up_csv)
print(f"rows: vendored {len(vend)}, upstream {len(up)}")
if "n_loaded" in up: print("upstream files loaded per folder:", up["n_loaded"].value_counts().to_dict())

# 1) per conformer, vendored vs upstream
key = "feather_dir"
report(vend.set_index(key), up.set_index(key), "per conformer: vendored vs upstream")

# 2) aggregated, each vs reference CSV
ref = pd.read_csv(ref_csv).set_index("inchi")
agg_v, agg_u = aggregate(vend), aggregate(up)
print(f"\nreference: {ref.shape[0]} rows, {ref.shape[1]} columns; DescriPyTor-derived columns present in reference: {sum(c in agg_v.columns for c in ref.columns)}")
report(ref, agg_v, "aggregated: reference (20260423) vs vendored re-run")
tab = report(ref, agg_u, "aggregated: reference (20260423) vs upstream re-run")
out = os.path.join(os.path.dirname(os.path.abspath(up_csv)), "column_diff_upstream_vs_reference.csv")
tab.sort_values("max_abs_diff", ascending=False).to_csv(out); print(f"\nper-column table -> {out}")
