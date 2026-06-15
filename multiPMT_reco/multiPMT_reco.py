import numpy as np
import pandas as pd
import iminuit
from iminuit.cost import LeastSquares
from iminuit import Minuit


ENERGY_RECO_PARAMS = {
    
    "H3FF": {"m": 429.74, "q": -6.507e3},
    "K3FF": {"m": 310.8,  "q": -5730.2},
    
                     }

def E_reco(PE, fit_type):
    
    try:
        
        m = FIT_PARAMS[fit_type]["m"]
        q = FIT_PARAMS[fit_type]["q"]
        
    except KeyError:
        raise ValueError(f"Invalid fit_type '{fit_type}'. Expected one of {list(FIT_PARAMS)}")
    
    return (PE - q) / m


def core_reco(pe_x_array: np.ndarray, pe_y_array: np.ndarray):

    df = pd.DataFrame({"x": pe_x_array, "y": pe_y_array})
    qmap = df.groupby(["x", "y"]).size().reset_index(name="q")
    
    if len(qmap) == 0:
        return np.nan, np.nan
    
    x = qmap["x"].to_numpy()
    y = qmap["y"].to_numpy()
    q = qmap["q"].to_numpy()

    threshold_cut = np.percentile(q, 80)
    q_cut = q[q > threshold_cut]
    x_cut = x[q > threshold_cut]
    y_cut = y[q > threshold_cut]
    
    x_weighted_above_cut = np.sum([i*j for i,j in zip(x_cut, q_cut)]) / np.sum(q_cut)
    y_weighted_above_cut = np.sum([i*j for i,j in zip(y_cut, q_cut)]) / np.sum(q_cut)

    sigma = np.sqrt(np.maximum(q, 1))

    def ldf_model(xy, xc, yc, A, r0, beta, bkg):
        x_data, y_data = xy
        r = np.sqrt((x_data - xc)**2 + (y_data - yc)**2)
        return A * (1 + r/r0)**(-beta) + bkg

    least_squares = LeastSquares((x, y), q, sigma, ldf_model)
    
    m = Minuit(least_squares, xc = x_weighted_above_cut, yc = y_weighted_above_cut, A = q.max(), r0 = 2e3, beta = 3, bkg = 0)

    m.limits["A"] = (0, None)
    m.limits["r0"] = (1, 1e5)
    m.limits["beta"] = (0.1, 10)
    m.limits["bkg"] = (0, None)

    m.strategy = 2  
    m.migrad()
    m.hesse()

    x_reco = m.values["xc"]
    y_reco = m.values["yc"]

    return x_reco, y_reco



def direction_reco(pe_x_array: np.ndarray, pe_y_array: np.ndarray, pe_times, x_reco : float, y_reco : float):

    c = 29.9792 #cm/ns

    def tfirst_model(xy, t0 : float, R : float, theta : float, phi : float):
        
        ux = np.sin(theta) * np.cos(phi)
        uy = np.sin(theta) * np.sin(phi)
        uz = np.cos(theta)
        pe_x_array, pe_y_array = xy  
        
        return t0 + (1/c) * np.sqrt((pe_x_array - x_reco + R*ux)**2 + (pe_y_array - y_reco + R*uy)**2 + (R*uz)**2)

    df = pd.DataFrame({"x": pe_x_array, "y": pe_y_array, "times": pe_times})
    tmap = df.groupby(["x", "y"])["times"].agg(times=list, t_first="min").reset_index()
    
    if tmap.empty:
        return np.nan, np.nan
    
    pe_x = tmap["x"].to_numpy()
    pe_y = tmap["y"].to_numpy()
    t_first = tmap["t_first"].to_numpy()
    
    if len(t_first) < 4:
        return np.nan, np.nan
    
    t_first = t_first - np.min(t_first)
    sigma = np.full_like(t_first, 1.0)
    
    least_squares = LeastSquares((pe_x, pe_y), t_first, sigma, tfirst_model)

    R_init = 1e4
    theta_init = 0.3
    phi_init = 0.0
    
    m = Minuit(least_squares, t0=0, R=R_init, theta=theta_init, phi=phi_init)

    m.limits["R"] = (1e2, 1e6)
    m.limits["theta"] = (1e-3, np.pi/2)
    m.limits["phi"] = (-np.pi, np.pi)

    m.strategy = 2  
    m.migrad()
    m.hesse()

    
    theta_reco = m.values["theta"]
    phi_reco = m.values["phi"]
    phi_reco = (phi_reco + np.pi) % (2*np.pi)
    
    return theta_reco, phi_reco


def PE_calibration(df: pd.DataFrame):

    df_dummy = df.copy()

    for i in range(7):
        ch = f"ch{i}"
        df_dummy[ch] = df_dummy[ch].apply(lambda x: 0.7 * np.asarray(x))

    return df_dummy