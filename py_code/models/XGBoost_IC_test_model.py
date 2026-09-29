from pathlib import Path
import pandas as pd
import joblib


def test(path: str, model_save_path: str, min_st_dist: float = 0, max_st_dist: float = 700, pe_min: float = 1, traces = False):

    df = pd.read_parquet(path)
    df = df[(df["T_C"] >= min_st_dist) & (df["T_C"] <= max_st_dist) & (df["total_pe"] >= pe_min)]

    if df.empty:
        print(f"WARNING: Empty dataset for {path}, skipping...")
        return None, None, None

    info_columns = ["T_C", "Nom_Th", "Nom_En", "Nom_Y_C", "Nom_X_C", "R_T", "X_T", "Y_T"]
    info = df[info_columns].copy()

    df = df.drop(columns=info_columns[1:])

    if not traces:
        X_te = df.drop(columns=["IsThereMuon", "ch_0", "ch_ref", "ch_60", "ch_120", "ch_180", "ch_240", "ch_300"], errors="ignore")
    else:
        X_te = df.drop(columns=["IsThereMuon"])

    y_te = df["IsThereMuon"].to_numpy()

    model = joblib.load(model_save_path)
    p_te = model.predict_proba(X_te)[:, 1]

    return y_te, p_te, info