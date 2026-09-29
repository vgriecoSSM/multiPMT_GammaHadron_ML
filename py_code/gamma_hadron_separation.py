from typing import NamedTuple, Sequence
from pathlib import Path

from matplotlib.legend_handler import HandlerBase
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.patches import Rectangle
import matplotlib.transforms as mtransforms
from numpy.typing import ArrayLike
import numpy as np
from iminuit import Minuit


class OffsetLineHandler(HandlerLine2D):

    def __init__(self, x_offset=0, y_offset=0, **kwargs):
        self.x_offset = x_offset
        self.y_offset = y_offset
        super().__init__(**kwargs)


    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):

        artists = super().create_artists(legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)
        offset_transform = mtransforms.Affine2D().translate(self.x_offset, self.y_offset)

        for artist in artists:
            artist.set_transform(artist.get_transform() + offset_transform)

        return artists


class OffsetHandler(HandlerBase):

    def __init__(self, x_offset=0, y_offset=0, **kwargs):
        self.x_offset = x_offset
        self.y_offset = y_offset
        super().__init__(**kwargs)


    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):

        patch = Rectangle((0, 0), width, height, facecolor=orig_handle.get_facecolor(), edgecolor=orig_handle.get_edgecolor(), linewidth=orig_handle.get_linewidth())
        offset_transform = trans + mtransforms.Affine2D().translate(self.x_offset, self.y_offset)
        patch.set_transform(offset_transform)

        return [patch]


class Triggered_Primaries(NamedTuple):

    triggered_protons: list
    triggered_gammas: list


class PrimaryCounterResult(NamedTuple):

    protons_info_val: list
    protons_info_test: list
    number_of_protons: int
    
    gammas_info_val: list
    gammas_info_test: list
    number_of_gammas: int

class Threshold_Hunter(NamedTuple):

    best_threshold_val: float
    minimum_fpr_on_validation : float 
    

class Summed_Probabilities(NamedTuple):
    
    sum_probs_protons_val: list
    sum_probs_protons_test: list
    
    sum_probs_gammas_val: list
    sum_probs_gammas_test: list

class Mean_Summed_Probabilities(NamedTuple):
    
    mean_sum_probs_protons_val: list
    mean_sum_probs_protons_test: list
    
    mean_sum_probs_gammas_val: list
    mean_sum_probs_gammas_test: list

class S_to_B(NamedTuple):
    
    S_list: list
    S_to_sqrtB_list: list

class ROCPoints(NamedTuple):
    
    tpr: list
    fpr: list
    thresholds: list


class MuonProbabilityData(NamedTuple):

    n_muons: np.ndarray
    p_cum: np.ndarray
    bin_centers: np.ndarray
    mean_pr: np.ndarray
    err_pr: np.ndarray
    p_bin_centers: list[np.ndarray]

class LinearFitResults(NamedTuple):

    slope: float
    slope_error: float
    intercept: float
    intercept_error: float
    chi2: float
    ndf: int
    rho: float
    n_fit: np.ndarray
    p_fit: np.ndarray
    fit_valid: bool


class EstimatorResults(NamedTuple):

    est_list: list[np.ndarray]
    resolution: np.ndarray
    bias: np.ndarray
    bias_error: np.ndarray


class MuonResidualResults(NamedTuple):

    delta_n: np.ndarray
    bias: float
    p16: float
    p84: float
    width_68: float



def station_multiplicity_trigger(protons : list, gammas: list, multiplicity : int = 1):

    triggered_protons = []
    triggered_gammas = []

    for proton in protons:
        
        p_proton = np.asarray(proton[0], dtype = float)

        n_stations_proton = len(p_proton)
        
        if n_stations_proton >= multiplicity:

            triggered_protons.append(proton)

    for gamma in gammas:

        p_gammas = np.asarray(gamma[0], dtype = float)

        n_stations_gammas = len(p_gammas)

        if n_stations_gammas >= multiplicity:

            triggered_gammas.append(gamma)

    return Triggered_Primaries(triggered_protons = triggered_protons, triggered_gammas = triggered_gammas)

        
def get_primary_results(proton_txt_dir : str, gamma_txt_dir : str, gamma_hadron_discrimination_val_split : float = 0.5):
    
    proton_files = list(proton_txt_dir.glob("*.txt"))
    gamma_files  = list(gamma_txt_dir.glob("*.txt"))
    
    n_proton_val = int(gamma_hadron_discrimination_val_split * len(proton_files))
    n_gamma_val  = int(gamma_hadron_discrimination_val_split * len(gamma_files))

    protons_info_val = []
    gammas_info_val = []
    
    protons_info_test = []
    gammas_info_test = []
    
    p_val_count = 0
    g_val_count = 0
    
    number_of_protons = 0
    number_of_gammas = 0
    
    for txt_file in proton_txt_dir.glob("*.txt"):
        name = txt_file.stem   
        txt_file_name = str(txt_file)
        y_te, p_te = np.loadtxt(txt_file_name, usecols=(0, 1), ndmin=2).T 
    
        if p_val_count < n_proton_val:
            
                protons_info_val.append((p_te.tolist(),y_te.tolist()))
                p_val_count += 1
        else:
                protons_info_test.append((p_te.tolist(),y_te.tolist()))
    
        number_of_protons += 1
    
    for txt_file in gamma_txt_dir.glob("*.txt"):
        name = txt_file.stem   
        txt_file_name = str(txt_file)
        y_te, p_te = np.loadtxt(txt_file_name, usecols=(0, 1), ndmin=2).T   
    
        if g_val_count < n_gamma_val:
            
            gammas_info_val.append((p_te.tolist(),y_te.tolist()))
            g_val_count += 1
            
        else:
            gammas_info_test.append((p_te.tolist(),y_te.tolist())) 
    
        number_of_gammas += 1

    return PrimaryCounterResult(protons_info_val = protons_info_val,
                                       protons_info_test = protons_info_test,
                                       number_of_protons = len(proton_files),
                                       gammas_info_val = gammas_info_val,
                                       gammas_info_test = gammas_info_test,
                                       number_of_gammas = len(gamma_files)
                                       )

def get_best_threshold_on_val(protons_info_val :  list, gammas_info_val : list, target_eff: float = 0.8):

    fprs = []
    thresholds = np.linspace(0, 1, 1000)
    
    for t in thresholds:

        sum_probs_protons = []
        sum_probs_gammas = []
        
        for i in range(len(protons_info_val)):
            
            info_single_proton = protons_info_val[i]

            p_proton = np.asarray(info_single_proton[0])
            y_proton = np.asarray(info_single_proton[1])
    
            p_proton_above_cut = np.sum(p_proton[p_proton >= t])
        
            sum_probs_protons.append(p_proton_above_cut)
            

        for i in range(len(gammas_info_val)):
            
            info_single_gamma = gammas_info_val[i]

            p_gamma = np.asarray(info_single_gamma[0])
            y_gamma = np.asarray(info_single_gamma[1])
    
            p_gamma_above_cut = np.sum(p_gamma[p_gamma >= t])
        
            sum_probs_gammas.append(p_gamma_above_cut)

    
        thresh = np.percentile(sum_probs_gammas, target_eff * 100)
    
        proton_fpr = np.sum(sum_probs_protons <= thresh) / len(sum_probs_protons)
    
        fprs.append(proton_fpr)

    best_index = np.nanargmin(fprs)
    best_threshold_val = thresholds[best_index]
    minimum_fpr_on_validation = fprs[best_index]

    print(f"Gamma efficiency target = {target_eff}")
    print(f"Min FPR on Validation = {minimum_fpr_on_validation:.2e}")
    print(f"Best Threshold on Validation = {best_threshold_val:2f}")

    return Threshold_Hunter(best_threshold_val = best_threshold_val, minimum_fpr_on_validation = minimum_fpr_on_validation)


def get_summed_probabilities(best_threshold_val : float, protons_validation : list, protons_testing: list, gammas_validation: list, gammas_testing: list):
    
    sum_probs_protons_test = []
    sum_probs_gammas_test = []
    sum_probs_protons_val = []
    sum_probs_gammas_val = []
    
    for i in range(len(protons_testing)):
        info_single_proton_test = protons_testing[i]
        p_proton_test = np.asarray(info_single_proton_test[0])
        p_proton_above_cut_test = np.sum(p_proton_test[p_proton_test >= best_threshold_val])   
        sum_probs_protons_test.append(p_proton_above_cut_test)
    
    for i in range(len(gammas_testing)):
        info_single_gamma_test = gammas_testing[i]
        p_gamma_test = np.asarray(info_single_gamma_test[0])
        p_gamma_above_cut_test = np.sum(p_gamma_test[p_gamma_test >= best_threshold_val])   
        sum_probs_gammas_test.append(p_gamma_above_cut_test)
    
    for i in range(len(protons_validation)):
        info_single_proton_val = protons_validation[i]
        p_proton_val = np.asarray(info_single_proton_val[0])
        p_proton_above_cut_val = np.sum(p_proton_val[p_proton_val >= best_threshold_val])   
        sum_probs_protons_val.append(p_proton_above_cut_val)
    
    for i in range(len(gammas_validation)):
        info_single_gamma_val = gammas_validation[i]
        p_gamma_val = np.asarray(info_single_gamma_val[0])
        p_gamma_above_cut_val = np.sum(p_gamma_val[p_gamma_val >= best_threshold_val])   
        sum_probs_gammas_val.append(p_gamma_above_cut_val)

    return Summed_Probabilities(sum_probs_protons_val = sum_probs_protons_val,
                                sum_probs_protons_test = sum_probs_protons_test,
                                sum_probs_gammas_val = sum_probs_gammas_val,
                                sum_probs_gammas_test = sum_probs_gammas_test
                                )

def get_best_threshold_on_val_mean(protons_info_val : list, gammas_info_val : list, target_eff : float = 0.8):

    fprs = []
    thresholds = np.linspace(0, 1, 1000)

    for t in thresholds:

        mean_sum_probs_protons = []
        mean_sum_probs_gammas = []

        for i in range(len(protons_info_val)):

            info_single_proton = protons_info_val[i]
            p_proton = np.asarray(info_single_proton[0], dtype = float)

            mean_p_proton_above_cut = np.sum(p_proton[p_proton >= t]) / len(p_proton) if len(p_proton) > 0 else 0.0

            mean_sum_probs_protons.append(mean_p_proton_above_cut)

        for i in range(len(gammas_info_val)):

            info_single_gamma = gammas_info_val[i]
            p_gamma = np.asarray(info_single_gamma[0], dtype = float)

            mean_p_gamma_above_cut = np.sum(p_gamma[p_gamma >= t]) / len(p_gamma) if len(p_gamma) > 0 else 0.0

            mean_sum_probs_gammas.append(mean_p_gamma_above_cut)

        mean_sum_probs_protons = np.asarray(mean_sum_probs_protons, dtype = float)
        mean_sum_probs_gammas = np.asarray(mean_sum_probs_gammas, dtype = float)

        thresh = np.percentile(mean_sum_probs_gammas, target_eff * 100)

        proton_fpr = np.mean(mean_sum_probs_protons <= thresh)

        fprs.append(proton_fpr)

    best_index = np.nanargmin(fprs)
    best_threshold_val = thresholds[best_index]
    minimum_fpr_on_validation = fprs[best_index]

    print(f"Gamma efficiency target = {target_eff}")
    print(f"Min FPR on Validation = {minimum_fpr_on_validation:.2e}")
    print(f"Best Threshold on Validation = {best_threshold_val:.6f}")

    return Threshold_Hunter(best_threshold_val = best_threshold_val, minimum_fpr_on_validation = minimum_fpr_on_validation)



def get_mean_summed_probabilities(best_threshold_val : float, protons_validation : list, protons_testing : list, gammas_validation : list, gammas_testing : list):

    mean_sum_probs_protons_test = []
    mean_sum_probs_gammas_test = []
    mean_sum_probs_protons_val = []
    mean_sum_probs_gammas_val = []

    n_tank_protons = []
    n_tank_gammas = []

    for i in range(len(protons_testing)):
        
        info_single_proton_test = protons_testing[i]
        p_proton_test = np.asarray(info_single_proton_test[0], dtype = float)

        mean_p_proton_above_cut_test = np.sum(p_proton_test[p_proton_test >= best_threshold_val]) / len(p_proton_test) if len(p_proton_test) > 0 else 0.0

        mean_sum_probs_protons_test.append(mean_p_proton_above_cut_test)

        if len(p_proton_test) == 0:
                print("WARNING: 0 tanks fired!, invalid event")
            
        n_tank_protons.append(len(p_proton_test))

    print("Mean Number of fired tanks for Protons: ", np.mean(n_tank_protons))

    for i in range(len(gammas_testing)):
        info_single_gamma_test = gammas_testing[i]
        p_gamma_test = np.asarray(info_single_gamma_test[0], dtype = float)

        mean_p_gamma_above_cut_test = np.sum(p_gamma_test[p_gamma_test >= best_threshold_val]) / len(p_gamma_test) if len(p_gamma_test) > 0 else 0.0

        mean_sum_probs_gammas_test.append(mean_p_gamma_above_cut_test)
        
        n_tank_gammas.append(len(p_gamma_test))
        
        if len(p_gamma_test) == 0:
                print("WARNING: 0 tanks fired!, invalid event")
        
    print("Mean Number of fired tanks for Gammas: ", np.mean(n_tank_gammas))

    for i in range(len(protons_validation)):
        info_single_proton_val = protons_validation[i]
        p_proton_val = np.asarray(info_single_proton_val[0], dtype = float)

        mean_p_proton_above_cut_val = np.sum(p_proton_val[p_proton_val >= best_threshold_val]) / len(p_proton_val) if len(p_proton_val) > 0 else 0.0

        mean_sum_probs_protons_val.append(mean_p_proton_above_cut_val)

    for i in range(len(gammas_validation)):
        info_single_gamma_val = gammas_validation[i]
        p_gamma_val = np.asarray(info_single_gamma_val[0], dtype = float)

        mean_p_gamma_above_cut_val = np.sum(p_gamma_val[p_gamma_val >= best_threshold_val]) / len(p_gamma_val) if len(p_gamma_val) > 0 else 0.0

        mean_sum_probs_gammas_val.append(mean_p_gamma_above_cut_val)

    return Mean_Summed_Probabilities(
        mean_sum_probs_protons_val = mean_sum_probs_protons_val,
        mean_sum_probs_protons_test = mean_sum_probs_protons_test,
        mean_sum_probs_gammas_val = mean_sum_probs_gammas_val,
        mean_sum_probs_gammas_test = mean_sum_probs_gammas_test
    )

def get_S_to_B_ratio(gammas: np.ndarray, protons: np.ndarray):
    
    S_list = []
    B_list = []
    S_to_sqrtB_list = []

    for percentile in np.linspace(0, 1, 20):
        thresh = np.percentile(gammas, percentile*100)  

        S = np.sum(gammas <= thresh) / len(gammas)
        B = np.sum(protons <= thresh) / len(protons)
        
        S_list.append(S)
        B_list.append(B)

        if B > 0:
            S_to_sqrtB_list.append(S / np.sqrt(B))
        else:
            S_to_sqrtB_list.append(np.nan)

    return S_to_B(S_list = S_list, S_to_sqrtB_list = S_to_sqrtB_list)


def get_ROC(gammas : np.ndarray, protons : np.ndarray, n_points : int = 200) -> ROCPoints:

    gammas = np.asarray(gammas, dtype = float)
    protons = np.asarray(protons, dtype = float)
    
    gammas = gammas[np.isfinite(gammas)]
    protons = protons[np.isfinite(protons)]

    thresholds = np.linspace(min(gammas.min(), protons.min()), max(gammas.max(), protons.max()), n_points)
    tpr = []
    fpr = []

    for threshold in thresholds:

        tpr.append(np.mean(gammas <= threshold))
        fpr.append(np.mean(protons <= threshold))

    tpr = np.asarray(tpr, dtype=float)
    fpr = np.asarray(fpr, dtype=float)

    return ROCPoints(tpr = tpr, fpr = fpr, thresholds = thresholds)



def save_ROC_results(results_save_dir, validation_split, protons_val, gammas_val, protons_test, gammas_test, threshold_val, gamma_fraction_val, proton_fraction_val, gamma_fraction_test, proton_fraction_test, tpr_val, fpr_val, thr_val, tpr_test, fpr_test, thr_test):

    results_save_dir = Path(results_save_dir)
    results_save_dir.mkdir(parents=True, exist_ok=True)
    legend_info_txt = results_save_dir / f"{validation_split}val_legend_info.txt"
    roc_points_txt = results_save_dir / f"{validation_split}val_roc_points.txt"

    with open(legend_info_txt, "w") as f:

        f.write("ROC legend information\n")
        f.write("======================\n\n")
        f.write("Validation curve\n")
        f.write(f"Proton entries = {len(protons_val)}\n")
        f.write(f"Gamma entries = {len(gammas_val)}\n\n")
        f.write("Test curve\n")
        f.write(f"Proton entries = {len(protons_test)}\n")
        f.write(f"Gamma entries = {len(gammas_test)}\n\n")
        f.write("Validation cut applied to validation\n")
        f.write(f"Validation threshold = {threshold_val:.8f}\n")
        f.write(f"Gamma efficiency = {gamma_fraction_val:.8f}\n")
        f.write(f"Proton fraction / FPR = {proton_fraction_val:.8e}\n\n")
        f.write("Validation cut applied to test\n")
        f.write(f"Validation threshold = {threshold_val:.8f}\n")
        f.write(f"Gamma efficiency = {gamma_fraction_test:.8f}\n")
        f.write(f"Proton fraction / FPR = {proton_fraction_test:.8e}\n")

    n_val = len(tpr_val)
    n_test = len(tpr_test)
    n_max = max(n_val, n_test)

    with open(roc_points_txt, "w") as f:

        f.write("# ROC points to reproduce validation and test curves\n")
        f.write("# Columns:\n")
        f.write("# index tpr_val fpr_val threshold_val_curve tpr_test fpr_test threshold_test_curve\n")

        for i in range(n_max):

            if i < n_val:

                val_str = f"{tpr_val[i]:.10e} {fpr_val[i]:.10e} {thr_val[i]:.10e}"

            else:

                val_str = "nan nan nan"

            if i < n_test:

                test_str = f"{tpr_test[i]:.10e} {fpr_test[i]:.10e} {thr_test[i]:.10e}"

            else:

                test_str = "nan nan nan"

            f.write(f"{i} {val_str} {test_str}\n")

    return 0


def get_probabilities_and_nmu(protons_test : Sequence[tuple[ArrayLike, ArrayLike]], best_threshold_val : float, bin_width : int = 5, min_bin_entries: int = 20):
    
    n_muons = []
    p_cum = []

    for info_single_proton in protons_test:

        p_proton = np.asarray(info_single_proton[0], dtype=float)
        y_proton = np.asarray(info_single_proton[1], dtype=float)

        mask_thr = p_proton >= best_threshold_val

        p_proton_above_threshold = p_proton[mask_thr]
        n_muons_proton = np.nansum(y_proton == 1)
        p_cum_proton = np.nansum(p_proton_above_threshold)
        n_muons.append(n_muons_proton)
        p_cum.append(p_cum_proton)

    n_muons = np.asarray(n_muons, dtype=float)
    p_cum = np.asarray(p_cum, dtype=float)


    n_min = np.floor(np.min(n_muons) / bin_width) * bin_width
    n_max = (np.floor(np.max(n_muons) / bin_width) + 1) * bin_width
    bins = np.arange(n_min, n_max + bin_width, bin_width)
    
    bin_centers = []
    mean_pr = []
    err_pr = []
    p_bin_centers = []

    for i in range(len(bins) - 1):

        mask_bin = (n_muons >= bins[i]) & (n_muons < bins[i + 1])
        p_cum_bin = p_cum[mask_bin]

        if len(p_cum_bin) > min_bin_entries:

            bin_centers.append(0.5 * (bins[i] + bins[i + 1]))
            mean_pr.append(np.mean(p_cum_bin))
            err_pr.append(np.std(p_cum_bin) / np.sqrt(len(p_cum_bin)))
            p_bin_centers.append(p_cum_bin)

    bin_centers = np.asarray(bin_centers, dtype=float)
    mean_pr = np.asarray(mean_pr, dtype=float)
    err_pr = np.asarray(err_pr, dtype=float)

    return MuonProbabilityData(n_muons = n_muons, p_cum = p_cum, bin_centers = bin_centers, mean_pr = mean_pr, err_pr = err_pr, p_bin_centers = p_bin_centers)


def get_mean_probabilities_and_nmu(protons_test : Sequence[tuple[ArrayLike, ArrayLike]], best_threshold_val : float, bin_width : int = 5, min_bin_entries : int = 20):

    n_muons = []
    p_cum = []

    for info_single_proton in protons_test:

        p_proton = np.asarray(info_single_proton[0], dtype=float)
        y_proton = np.asarray(info_single_proton[1], dtype=float)

        mask_thr = p_proton >= best_threshold_val

        p_proton_above_threshold = p_proton[mask_thr]
        n_muons_proton = np.nansum(y_proton == 1) / len(p_proton) if len(p_proton) > 0 else 0.0
        p_cum_proton = np.nansum(p_proton_above_threshold) / len(p_proton) if len(p_proton) > 0 else 0.0

        n_muons.append(n_muons_proton)
        p_cum.append(p_cum_proton)

    n_muons = np.asarray(n_muons, dtype=float)
    p_cum = np.asarray(p_cum, dtype=float)

    n_min = np.floor(np.min(n_muons) / bin_width) * bin_width
    n_max = (np.floor(np.max(n_muons) / bin_width) + 1) * bin_width
    bins = np.arange(n_min, n_max + bin_width, bin_width)

    bin_centers = []
    mean_pr = []
    err_pr = []
    p_bin_centers = []

    for i in range(len(bins) - 1):

        mask_bin = (n_muons >= bins[i]) & (n_muons < bins[i + 1])
        p_cum_bin = p_cum[mask_bin]

        if len(p_cum_bin) > min_bin_entries:

            bin_centers.append(0.5 * (bins[i] + bins[i + 1]))
            mean_pr.append(np.mean(p_cum_bin))
            err_pr.append(np.std(p_cum_bin) / np.sqrt(len(p_cum_bin)))
            p_bin_centers.append(p_cum_bin)

    bin_centers = np.asarray(bin_centers, dtype=float)
    mean_pr = np.asarray(mean_pr, dtype=float)
    err_pr = np.asarray(err_pr, dtype=float)

    return MuonProbabilityData(n_muons = n_muons, p_cum = p_cum, bin_centers = bin_centers, mean_pr = mean_pr, err_pr = err_pr, p_bin_centers = p_bin_centers)



def fit_probabilities_vs_nmu(bin_centers : ArrayLike, mean_pr : ArrayLike, err_pr : ArrayLike, n_muons : ArrayLike, p_cum : ArrayLike, n_fit_min : float = 0, n_fit_max : float = 100, n_fit_points : int = 300) -> LinearFitResults:

    bin_centers = np.asarray(bin_centers, dtype=float)
    mean_pr = np.asarray(mean_pr, dtype=float)
    err_pr = np.asarray(err_pr, dtype=float)
    n_muons = np.asarray(n_muons, dtype=float)
    p_cum = np.asarray(p_cum, dtype=float)

    valid_fit_points = np.isfinite(bin_centers) & np.isfinite(mean_pr) & np.isfinite(err_pr) & (err_pr > 0)
    bin_centers_fit = bin_centers[valid_fit_points]
    mean_pr_fit = mean_pr[valid_fit_points]
    err_pr_fit = err_pr[valid_fit_points]

    if len(bin_centers_fit) < 3:

        raise ValueError("Not enough valid points to perform the linear fit.")

    def linear_regr(x, a, b):

        return a * x + b


    def chi2(a, b):

        return np.sum(((mean_pr_fit - linear_regr(bin_centers_fit, a, b)) / err_pr_fit) ** 2)


    m = Minuit(chi2, a=1.0, b=0.0)
    m.errordef = Minuit.LEAST_SQUARES
    m.migrad()
    m.hesse()

    slope = float(m.values["a"])
    slope_error = float(m.errors["a"])
    intercept = float(m.values["b"])
    intercept_error = float(m.errors["b"])
    chi2_val = float(m.fval)
    ndf = len(mean_pr_fit) - len(m.parameters)
    
    n_fit = np.linspace(n_fit_min, n_fit_max, n_fit_points)
    p_fit = linear_regr(n_fit, slope, intercept)

    valid_correlation = np.isfinite(n_muons) & np.isfinite(p_cum)

    rho = np.corrcoef(n_muons[valid_correlation], p_cum[valid_correlation])[0, 1]

    return LinearFitResults(slope=slope, slope_error=slope_error, intercept=intercept, intercept_error=intercept_error, chi2=chi2_val, ndf=ndf, rho=rho, n_fit=n_fit, p_fit=p_fit, fit_valid=m.valid)
    

def get_estimator_results(bin_centers : ArrayLike, p_bin_centers : Sequence[ArrayLike], slope : float, intercept : float):

    bin_centers = np.asarray(bin_centers, dtype=float)

    def n_estimated(p):

        return (p - intercept) / slope


    def estimator(n_reco, n_muons):

        return 1 - n_reco / n_muons


    est_list = []

    for i in range(len(bin_centers)):

        if bin_centers[i] == 0:

            est = np.full(len(p_bin_centers[i]), np.nan)

        else:

            p_values = np.asarray(p_bin_centers[i], dtype=float)
            n_reco = n_estimated(p_values)
            est = estimator(n_reco, bin_centers[i])
            
        est_list.append(est)

    resolution = np.asarray([(np.nanpercentile(values, 84) - np.nanpercentile(values, 16)) / 2 for values in est_list])
    bias = np.asarray([np.nanmean(values) for values in est_list])

    bias_error = np.asarray([np.nanstd(values, ddof=1) / np.sqrt(np.sum(np.isfinite(values))) if np.sum(np.isfinite(values)) > 1 else np.nan for values in est_list])

    return EstimatorResults(est_list = est_list, resolution = resolution, bias = bias, bias_error = bias_error)


def get_muon_residual_results(protons_test : Sequence[tuple[ArrayLike, ArrayLike]], best_threshold : float, slope : float, intercept : float):

    delta_n = []

    for info_single_proton in protons_test:

        p_proton = np.asarray(info_single_proton[0], dtype=float)
        y_proton = np.asarray(info_single_proton[1], dtype=float)
        p_proton = p_proton[p_proton >= best_threshold]
        n_true = np.nansum(y_proton == 1)
        p_cum = np.nansum(p_proton)

        if n_true > 0:

            n_reco = (p_cum - intercept) / slope
            delta_n.append((n_reco - n_true) / n_true)

    delta_n = np.asarray(delta_n, dtype=float)
    delta_n = delta_n[np.isfinite(delta_n)]

    bias = float(np.mean(delta_n))
    p16 = float(np.percentile(delta_n, 16))
    p84 = float(np.percentile(delta_n, 84))
    width_68 = (p84 - p16) / 2

    return MuonResidualResults(delta_n = delta_n, bias = bias, p16 = p16, p84 = p84, width_68 = width_68)

def get_mean_muon_residual_results(protons_test : Sequence[tuple[ArrayLike, ArrayLike]], best_threshold : float, slope : float, intercept : float):

    delta_n = []

    for info_single_proton in protons_test:

        p_proton = np.asarray(info_single_proton[0], dtype=float)
        y_proton = np.asarray(info_single_proton[1], dtype=float)

        n_stations = len(p_proton)
        n_true = np.nansum(y_proton == 1) / n_stations if n_stations > 0 else 0.0
    
        p_proton = p_proton[p_proton >= best_threshold]
        p_cum = np.nansum(p_proton) / n_stations if n_stations > 0 else 0.0

        if n_true > 0:

            n_reco = (p_cum - intercept) / slope
            delta_n.append((n_reco - n_true) / n_true)

    delta_n = np.asarray(delta_n, dtype=float)
    delta_n = delta_n[np.isfinite(delta_n)]

    bias = float(np.mean(delta_n))
    p16 = float(np.percentile(delta_n, 16))
    p84 = float(np.percentile(delta_n, 84))
    width_68 = (p84 - p16) / 2

    return MuonResidualResults(delta_n = delta_n, bias = bias, p16 = p16, p84 = p84, width_68 = width_68)