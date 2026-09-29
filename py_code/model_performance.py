import numpy as np

from pathlib import Path
from typing import NamedTuple
from numpy.typing import ArrayLike
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as calculate_auc


class FeatureImportanceData(NamedTuple):

    features: np.ndarray
    values: np.ndarray

class StationData(NamedTuple):

    y: np.ndarray
    probabilities: np.ndarray

class StationROCResults(NamedTuple):

    p_mu: np.ndarray
    p_bkg: np.ndarray
    p_mu_sm: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc: float
    fpr_smonly: np.ndarray
    tpr_smonly: np.ndarray
    thresholds_smonly: np.ndarray
    auc_smonly: float


class StationPRResults(NamedTuple):

    precision: np.ndarray
    recall: np.ndarray
    thresholds: np.ndarray
    pr_auc: float
    baseline: float
    precision_smonly: np.ndarray
    recall_smonly: np.ndarray
    thresholds_smonly: np.ndarray
    pr_auc_smonly: float
    baseline_smonly: float

def get_all_stations(txt_dir : str | Path) -> StationData:

    txt_dir = Path(txt_dir)
    txt_files = sorted(txt_dir.glob("*.txt"))

    if len(txt_files) == 0:
        raise FileNotFoundError(f"No txt files found in {txt_dir}")

    y_values = []
    probability_values = []

    for txt_file in txt_files:

        data = np.loadtxt(txt_file, ndmin=2)


        y_values.append(data[:, 0])
        probability_values.append(data[:, 1])

    y_values = np.concatenate(y_values)
    probability_values = np.concatenate(probability_values)

    return StationData(y = y_values, probabilities = probability_values)


def get_feature_importance(path : str | Path) -> FeatureImportanceData:

    path = Path(path)
    features = []
    values = []

    with open(path, "r") as file:

        for line in file:

            if line.strip():

                feature, value = line.strip().split()
                features.append(feature)
                values.append(float(value))

    return FeatureImportanceData(features = np.asarray(features), values = np.asarray(values, dtype=float))



def get_station_ROC(p_test : ArrayLike, y_test : ArrayLike, p_test_sm : ArrayLike, y_test_sm : ArrayLike) -> StationROCResults:

    p_test = np.asarray(p_test, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    p_test_sm = np.asarray(p_test_sm, dtype=float)
    y_test_sm = np.asarray(y_test_sm, dtype=float)
    valid_test = np.isfinite(p_test) & np.isfinite(y_test)
    valid_smonly = np.isfinite(p_test_sm) & np.isfinite(y_test_sm)
    p_test = p_test[valid_test]
    y_test = y_test[valid_test]
    p_test_sm = p_test_sm[valid_smonly]
    y_test_sm = y_test_sm[valid_smonly]
    p_mu = p_test[y_test == 1]
    p_bkg = p_test[y_test == 0]
    p_mu_sm = p_test_sm[y_test_sm == 1]



    y_roc = np.concatenate((np.ones(len(p_mu)), np.zeros(len(p_bkg))))
    p_roc = np.concatenate((p_mu, p_bkg))
    y_smonly_roc = np.concatenate((np.ones(len(p_mu_sm)), np.zeros(len(p_bkg))))
    p_smonly_roc = np.concatenate((p_mu_sm, p_bkg))
    fpr, tpr, thresholds = roc_curve(y_roc, p_roc)
    fpr_smonly, tpr_smonly, thresholds_smonly = roc_curve(y_smonly_roc, p_smonly_roc)
    auc = roc_auc_score(y_roc, p_roc)
    auc_smonly = roc_auc_score(y_smonly_roc, p_smonly_roc)

    return StationROCResults(p_mu = p_mu, p_bkg = p_bkg, p_mu_sm = p_mu_sm, fpr = fpr, tpr = tpr, thresholds = thresholds, auc = auc, fpr_smonly = fpr_smonly, tpr_smonly = tpr_smonly, thresholds_smonly = thresholds_smonly, auc_smonly = auc_smonly)


def get_station_PR(p_mu : ArrayLike, p_bkg : ArrayLike, p_mu_sm : ArrayLike):
    
    p_mu = np.asarray(p_mu, dtype=float)
    p_bkg = np.asarray(p_bkg, dtype=float)
    p_mu_sm = np.asarray(p_mu_sm, dtype=float)
    p_mu = p_mu[np.isfinite(p_mu)]
    p_bkg = p_bkg[np.isfinite(p_bkg)]
    p_mu_sm = p_mu_sm[np.isfinite(p_mu_sm)]

    y_roc = np.concatenate((np.ones(len(p_mu)), np.zeros(len(p_bkg))))
    p_roc = np.concatenate((p_mu, p_bkg))
    y_smonly_roc = np.concatenate((np.ones(len(p_mu_sm)), np.zeros(len(p_bkg))))
    p_smonly_roc = np.concatenate((p_mu_sm, p_bkg))
    precision, recall, thresholds = precision_recall_curve(y_roc, p_roc)
    precision_smonly, recall_smonly, thresholds_smonly = precision_recall_curve(y_smonly_roc, p_smonly_roc)
    pr_auc = calculate_auc(recall, precision)
    pr_auc_smonly = calculate_auc(recall_smonly, precision_smonly)

    baseline = np.mean(y_roc)
    baseline_smonly = np.mean(y_smonly_roc)

    return StationPRResults(precision = precision, recall = recall, thresholds = thresholds, pr_auc = pr_auc, baseline = baseline, precision_smonly = precision_smonly, recall_smonly = recall_smonly, thresholds_smonly = thresholds_smonly, pr_auc_smonly = pr_auc_smonly, baseline_smonly = baseline_smonly)

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


