import pandas as pd
import numpy as np
import math
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import optuna
from scipy.interpolate import interp1d
from scipy.stats import entropy
import joblib


def preprocess(path: str, traces=False, min_st_dist : float = 0, max_st_dist : float = 600, pe_min : float = 1):
    df = pd.read_parquet(path)
    df = df[df["T_C"] >= min_st_dist]
    df = df[df["T_C"] <= max_st_dist]
    df = df[df["total_pe"] >= pe_min]
    if not traces:
        X = df.drop(columns=['IsThereMuon', "ch_0", "ch_ref", "ch_60", "ch_120", "ch_180", "ch_240", "ch_300"], errors='ignore')
    else:
        X = df.drop(columns=['IsThereMuon'])
    y = df["IsThereMuon"]
    return X, y


def get_prauc_baseline(path: str):
    df = pd.read_parquet(path)
    print("Number of muons events = ", len(df[df["IsThereMuon"] == 1]))
    print("Number of NOT muons events = ", len(df[df["IsThereMuon"] == 0]))
    return df["IsThereMuon"].mean()


def xgb(trial, X, y, seed):
    scale_pos_weight = np.sum(y == 0) / np.sum(y == 1)
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000), #(1,1000)
        "max_depth": trial.suggest_int("max_depth", 3, 6), #(3,6)
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 3), #(2,3)
        "scale_pos_weight": scale_pos_weight, #scale_pos_weight,
        #"scale_pos_weight": trial.suggest_int("scale_pos_weight", 0, 25),
        "eval_metric": "aucpr",
        "random_state": seed
    }

    cv = StratifiedKFold(n_splits = 5, shuffle=True, random_state=seed)
    pr_aucs = []

    for tr, va in cv.split(X, y):
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("model", XGBClassifier(**params))
        ])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        p = pipe.predict_proba(X.iloc[va])[:, 1]
        pr, rc, _ = precision_recall_curve(y.iloc[va], p)
        pr_aucs.append(auc(rc, pr))

    return np.mean(pr_aucs)


def optuna_xgb(X, y, seed):
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: xgb(t, X, y, seed), n_trials = 15, show_progress_bar=True)
    best = study.best_params
    best.update({
        "scale_pos_weight": np.sum(y == 0) / np.sum(y == 1),
        "eval_metric": "aucpr",
        "random_state": seed
    })
    return best


def train(path: str, model_save_path: str, feat_imp_save_path: str, seed: int = 15, plot_imp = True, min_st_dist : float = 0, pe_min : float = 1, max_st_dist : float = 600):
    X, y = preprocess(path, min_st_dist = min_st_dist, pe_min = pe_min, max_st_dist = max_st_dist)
    outer_cv = StratifiedKFold(n_splits = 5, shuffle=True, random_state=seed)

    fold_scores = []
    fold_params = []

    for fold, (tr, va) in enumerate(outer_cv.split(X, y)):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        best_params = optuna_xgb(X_tr, y_tr, seed)
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("model", XGBClassifier(**best_params))
        ])
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_va)[:, 1]
        pr, rc, _ = precision_recall_curve(y_va, p)
        score = auc(rc, pr)

        fold_scores.append(score)
        fold_params.append(best_params)

    best_idx = int(np.argmax(fold_scores))
    final_params = fold_params[best_idx]

    final_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("model", XGBClassifier(**final_params))
    ])
    final_pipe.fit(X, y)
    joblib.dump(final_pipe, model_save_path)

    booster = final_pipe.named_steps["model"].get_booster()
    importance = booster.get_score(importance_type="gain")
    imp_series = pd.Series(importance)
    imp_series.index = [X.columns[int(k[1:])] for k in imp_series.index]
    imp_series = imp_series.sort_values(ascending=False)

    with open(feat_imp_save_path, "w") as f:
        for feature, value in imp_series.items():
            f.write(f"{feature} {value}\n")

    if plot_imp:
        plt.figure(figsize=(16, 9))
        plt.barh(imp_series.index, imp_series.values, color="black")
        plt.gca().invert_yaxis()
        plt.xlabel("Gain")
        plt.title("XGB Feature Importance")
        plt.tight_layout()
        plt.show()