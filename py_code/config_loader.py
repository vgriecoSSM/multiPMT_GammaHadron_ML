from pathlib import Path
import yaml


def load_config(config_path):
    config_path = Path(config_path)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    DAT_path_protons = cfg["paths"]["dat_path_protons"]
    DAT_path_gammas = cfg["paths"]["dat_path_gammas"]
    txt_tank_positions = cfg["paths"]["txt_tank_positions"]

    time_window = cfg["ranges"]["time_window_ns"]
    time_resolution = cfg["ranges"]["tdc_time_resolution_ns"]
    min_energy_reco = cfg["ranges"]["min_energy_reco_gev"]
    max_energy_reco = cfg["ranges"]["max_energy_reco_gev"]
    min_theta = cfg["ranges"]["min_theta_deg"]
    max_theta = cfg["ranges"]["max_theta_deg"]

    TRAIN_DAT_MIN = cfg["dat_ranges"]["train_dat_min"]
    TRAIN_DAT_MAX = cfg["dat_ranges"]["train_dat_max"]

    TEST_PROTONS_DAT_MIN = cfg["dat_ranges"]["test_protons_dat_min"]
    TEST_PROTONS_DAT_MAX = cfg["dat_ranges"]["test_protons_dat_max"]

    TEST_GAMMAS_DAT_MIN = cfg["dat_ranges"]["test_gammas_dat_min"]
    TEST_GAMMAS_DAT_MAX = cfg["dat_ranges"]["test_gammas_dat_max"]

    traces_dir = cfg["directories"]["traces_dir"]
    feats_dir = cfg["directories"]["feats_dir"]
    models_dir =  cfg["directories"]["models_dir"]
    output_dir = cfg["directories"]["output_dir"]
    results_dir = cfg["directories"]["results_dir"]
    pictures_dir = cfg["directories"]["pictures_dir"]

    array_number_of_channels = cfg["detector"]["array_number_of_channels"]
    number_of_channels_per_tank = cfg["detector"]["number_of_channels_per_tank"]
    FF = cfg["detector"]["ff"]
    tank_height = cfg["detector"]["tank_height_cm"]
    tank_radius = cfg["detector"]["tank_radius_cm"]
    intended_cut_on_pes = cfg["detector"]["intended_cut_on_pes"]
    tank_reflectiveness = cfg["detector"]["tank_reflectiveness"]
    mpmt_or_8inches = cfg["detector"]["mpmt_or_8inches"]
    tank_type = cfg["detector"]["tank_type"]
    ADC_MHz = cfg["detector"]["ADC_MHz"]
    ML_model = cfg["detector"]["model"]

    train_min_st_dist = cfg["training"]["train_min_st_dist_m"]
    train_max_st_dist = cfg["training"]["train_max_st_dist_m"]
    train_pe_min = cfg["training"]["train_pe_min"]

    test_min_st_dist = cfg["testing"]["test_min_st_dist_m"]
    test_max_st_dist = cfg["testing"]["test_max_st_dist_m"]
    test_pe_min = cfg["testing"]["test_pe_min"]

    gamma_hadron_discrimination_val_split = cfg["gamma_hadron_discrimination"]["validation_split"]
    event_stations_multiplicity = cfg["gamma_hadron_discrimination"]["event_stations_multiplicity"]
    FF_mask = cfg["gamma_hadron_discrimination"]["FF_mask"]

    array_dirs_raw = cfg["array_dirs"]

    array_dirs = {
        k.lower().strip(): v
        for k, v in array_dirs_raw.items()
    }

    key = mpmt_or_8inches.lower().strip()

    if key not in array_dirs:
        raise ValueError(
            f"Invalid mpmt_or_8inches = {mpmt_or_8inches!r}. "
            f"Allowed values are: {list(array_dirs_raw.keys())}"
        )

    array_dir = array_dirs[key]

    min_energy_tev = int(min_energy_reco / 1e3)
    max_energy_tev = int(max_energy_reco / 1e3)

    params_dir = (
        f"{min_energy_tev}{max_energy_tev}TeV_"
        f"{min_theta}{max_theta}deg_"
        f"{FF}FF_"
        f"{array_number_of_channels}channels_"
        f"{tank_radius}R{tank_height}h_"
        f"{tank_reflectiveness}"
    )

    time_resolution_label = str(time_resolution).replace(".", "p")

    cut_dir = (
        f"{intended_cut_on_pes}petrain_"
        f"{intended_cut_on_pes}petest_"
        f"{time_window}ns_"
        f"{time_resolution_label}nsTDC"
    )

    ADC_dir = f"{ADC_MHz}MHz_ADC" if ADC_MHz is not None else "NOADC"

    training_selection_dir = (
        f"{ML_model}_"
        f"train_dist_{int(train_min_st_dist):04d}_"
        f"{int(train_max_st_dist):04d}m_"
        f"{int(train_pe_min)}pe"
    )

    testing_selection_dir = f"test_dist_{test_min_st_dist:04d}_{test_max_st_dist:04d}m_{test_pe_min}pe_{event_stations_multiplicity}multiplicity_{FF_mask}maskFF"

    features_cut_dir = f"{ADC_dir}_{cut_dir}"

    
    return {
        "cfg": cfg,

        "DAT_path_protons": DAT_path_protons,
        "DAT_path_gammas": DAT_path_gammas,
        "txt_tank_positions": txt_tank_positions,

        "time_window": time_window,
        "time_resolution": time_resolution,
        "min_energy_reco": min_energy_reco,
        "max_energy_reco": max_energy_reco,
        "min_theta": min_theta,
        "max_theta": max_theta,

        "TRAIN_DAT_MIN": TRAIN_DAT_MIN,
        "TRAIN_DAT_MAX": TRAIN_DAT_MAX,

        "TEST_PROTONS_DAT_MIN": TEST_PROTONS_DAT_MIN,
        "TEST_PROTONS_DAT_MAX": TEST_PROTONS_DAT_MAX,

        "TEST_GAMMAS_DAT_MIN": TEST_GAMMAS_DAT_MIN,
        "TEST_GAMMAS_DAT_MAX": TEST_GAMMAS_DAT_MAX,

        "traces_dir": traces_dir,
        "feats_dir": feats_dir,
        "models_dir": models_dir,
        "output_dir": output_dir,
        "results_dir": results_dir,
        "pictures_dir": pictures_dir,

        "number_of_channels_per_tank": number_of_channels_per_tank,
        "array_number_of_channels": array_number_of_channels,
        "FF": FF,
        "tank_height": tank_height,
        "tank_radius": tank_radius,
        "intended_cut_on_pes": intended_cut_on_pes,
        "tank_reflectiveness": tank_reflectiveness,
        "mpmt_or_8inches": mpmt_or_8inches,
        "tank_type": tank_type,
        "ADC_MHz": ADC_MHz,
        "ML_model": ML_model,

        "train_min_st_dist": train_min_st_dist,
        "train_max_st_dist": train_max_st_dist,
        "train_pe_min": train_pe_min,

        "test_min_st_dist": test_min_st_dist,
        "test_max_st_dist": test_max_st_dist,
        "test_pe_min": test_pe_min,

        "array_dir": array_dir,
        "params_dir": params_dir,
        "cut_dir": cut_dir,
        "ADC_dir": ADC_dir,
        "features_cut_dir": features_cut_dir,
        "training_selection_dir": training_selection_dir,
        "testing_selection_dir" : testing_selection_dir,

        "gamma_hadron_discrimination_val_split" : gamma_hadron_discrimination_val_split,
        "event_stations_multiplicity" : event_stations_multiplicity,
        "FF_mask" : FF_mask
    }