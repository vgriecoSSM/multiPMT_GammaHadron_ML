import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from scipy.interpolate import interp1d
from scipy.stats import entropy
import joblib

def get_prauc_baseline(path: str):
    df = pd.read_parquet(path)
    return df["IsThereMuon"].mean()

def test(txt_name : str, path: str, model_save_path: str, output_results_path: str, min_st_dist : float = 0, max_st_dist : float = 600, pe_min : float = 1, traces = False, plot_pr = True):
    df = pd.read_parquet(path)
    df = df[df["T_C"] >= min_st_dist]
    df = df[df["T_C"] <= max_st_dist]
    df = df[df["total_pe"] >= pe_min]
    if not traces:
        X_te = df.drop(columns=['IsThereMuon', "ch_0", "ch_ref", "ch_60", "ch_120", "ch_180", "ch_240", "ch_300"], errors='ignore')
    else:
        X_te = df.drop(columns=['IsThereMuon'])
    y_te = df["IsThereMuon"]

    if X_te.shape[0] == 0:
        print(f"WARNING: Empty dataset for {txt_name}, skipping...")
        return None, None, None, None, None, None
    
    model = joblib.load(model_save_path)
    p_te = model.predict_proba(X_te)[:, 1]

    pr, rc, _ = precision_recall_curve(y_te, p_te)
    pr_auc = auc(rc, pr)
    baseline = get_prauc_baseline(path)

    n = max(len(rc), len(y_te))

    data = np.column_stack([pad(rc, n), pad(pr, n), pad(y_te, n), pad(p_te, n),])
    
    header = (f"PR_AUC={pr_auc:.3f}  Baseline={baseline:.3f}\n" "Recall Precision TrueLabel PredictedProbability")
    
    np.savetxt(output_results_path + txt_name, data, header=header, fmt="%.3f")

    if plot_pr:
        plt.figure(figsize=(7, 6))
        plt.plot(rc, pr, lw=2, label=f"PR AUC = {pr_auc:.3f}")
        plt.hlines(baseline, 0, 1, linestyles="dashed", label="Baseline")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return rc, pr, pr_auc, baseline, y_te, p_te

def pad(arr, n):
    
    out = np.full(n, np.nan)
    out[:len(arr)] = arr
    return out

def binned_inv_cdf(data, bins):
    
    counts, bin_edges = np.histogram(data, bins=bins, range=(0, 1))
    survival = np.cumsum(counts[::-1])[::-1] / len(data)
    x = bin_edges[:-1]
    survival = np.r_[survival, 0]
    x = np.r_[x, bin_edges[-1]]   
    return x, survival