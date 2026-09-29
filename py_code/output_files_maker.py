from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import pandas as pd


def load_tank_mask(tank_pos_file_path: str):

    tank_xy_m = np.loadtxt(tank_pos_file_path, usecols=(1, 2), ndmin = 2) / 100
    tank_xy_m = np.unique(tank_xy_m, axis = 0)

    return cKDTree(tank_xy_m)


def build_output_file(txt_name: str, output_results_path: str, y_te, p_te, info: pd.DataFrame, tank_type: str, FF_mask: str, tolerance_m: float = 0.1, ff_mask : float = 4):

    tank_pos_file_path = f"../survey_and_array_txt_repo/tank_pos_{tank_type}_{FF_mask}FF.txt"
    info_columns = ["T_C", "Nom_Th", "Nom_En", "Nom_Y_C", "Nom_X_C", "R_T", "X_T", "Y_T"]
    data = np.column_stack([y_te, p_te, info[info_columns].to_numpy()])

    tank_mask = load_tank_mask(tank_pos_file_path)
    distances, _ = tank_mask.query(info[["X_T", "Y_T"]].to_numpy())
    data = data[distances <= tolerance_m]

    output_dir = Path(output_results_path)
    output_dir.mkdir(parents = True, exist_ok = True)

    header = "TrueLabel PredictedProbability " + " ".join(info_columns)
    np.savetxt(output_dir / txt_name, data, header = header, fmt="%.10g")

    return data[:, 0].astype(int), data[:, 1]