from py_code import add_variables
from py_code import get_variables
from py_code import initialize_df
import numpy as np
import math
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq, gc

def build_df_feats_NOADC_mPMT(trace_path : str, save_path : str, mu_frac : float = 0, pe_threshold = 1, pe_lim = np.inf, ch0_present = False, max_time : int = 100):
    
    ADC_MHz = None
    df = initialize_df.get_df(path = trace_path, base = "pe", ADC_MHz = ADC_MHz, pe_threshold = pe_threshold, pe_lim = pe_lim)
    df = add_variables.add_time_Asimmetry_2ch(df = df, ch0_present = ch0_present)
    #df = add_variables.add_time_prompt_3ch(df = df, ch0_present = ch0_present)
    df = add_variables.add_time_Asimmetry_6ch(df = df, ch0_present = ch0_present)
    #df = add_variables.add_min_times(df = df)

    df = add_variables.add_mult_variables_to_df(df = df, get_var = get_variables.get_ch_pe)
    df = add_variables.add_mult_variables_to_df(df = df, get_var = get_variables.get_norm_ch_pe)
    df = add_variables.add_single_variable_to_df(df = df, get_var = get_variables.get_var_pe)

    df = add_variables.add_single_variable_to_df(df = df, get_var = get_variables.get_r_charge_bar, min_time = 0, max_time = max_time)
    #df = add_variables.add_single_variable_to_df(df = df, get_var = get_variables.get_x_charge_bar, min_time = 0, max_time = max_time)
    #df = add_variables.add_single_variable_to_df(df = df, get_var = get_variables.get_y_charge_bar, min_time = 0, max_time = max_time)
    
    df = add_variables.add_single_variable_to_df(df = df, get_var = get_variables.get_Asimmetry_6ch, min_time = 0, max_time = max_time, ch0_present = ch0_present)
    df = add_variables.add_single_variable_to_df(df = df, get_var = get_variables.get_Asimmetry_4ch, min_time = 0, max_time = max_time, ch0_present = ch0_present)
    df = add_variables.add_single_variable_to_df(df = df, get_var = get_variables.get_Asimmetry_2ch, min_time = 0, max_time = max_time, ch0_present = ch0_present)
    df = add_variables.add_single_variable_to_df(df = df, get_var = get_variables.get_Asimmetry_dynamic, min_time = 0, max_time = max_time, ch0_present = ch0_present)
    #df = add_variables.add_single_variable_to_df(df = df, get_var = get_variables.get_Asimmetry_60_240, min_time = 0, max_time = max_time, ch0_present = ch0_present)
    
    df = df.drop("ch_0", axis = 1)
    df = df.drop("ch_ref", axis = 1)
    df = df.drop("ch_60", axis = 1)
    df = df.drop("ch_120", axis = 1)
    df = df.drop("ch_180", axis = 1)
    df = df.drop("ch_240", axis = 1)
    df = df.drop("ch_300", axis = 1)

    #usually useless, only keeping T_C : distance relative to core
    df = df.drop("Nom_Th", axis = 1)
    df = df.drop("Nom_En", axis = 1) 
    df = df.drop("Nom_Y_C", axis = 1)
    df = df.drop("Nom_X_C", axis = 1)
    df = df.drop("R_T", axis = 1)
    df = df.drop("X_T", axis = 1)
    df = df.drop("Y_T", axis = 1)
    
    df = df[(df["IsThereMuon"] == 0) | ((df["IsThereMuon"] == 1) & (df["frac_mu"] >= mu_frac))]
    
    df_frac_mu_dropped = df.drop("frac_mu", axis=1)
    table_df_frac_mu_dropped = pa.Table.from_pandas(df_frac_mu_dropped)
    pq.write_table(table_df_frac_mu_dropped, save_path, compression="zstd")
    
    del df
    del df_frac_mu_dropped
    del table_df_frac_mu_dropped
    gc.collect()
    
    return None


def build_df_feats_multiPMT_withADC(trace_path : str, save_path : str, mu_frac : float = 0, pe_threshold = 1, pe_lim = np.inf, ADC_MHz: int = 1000, ch0_present: bool = False, max_time: int = 100):
    df = initialize_df.get_df(path=trace_path, base="pe", ADC_MHz=ADC_MHz, pe_threshold=pe_threshold, pe_lim=pe_lim)
    
    df = add_variables.add_time_Asimmetry_2ch(df=df, ch0_present=ch0_present)
    df = add_variables.add_time_prompt_3ch(df=df, ch0_present=ch0_present)
    df = add_variables.add_time_Asimmetry_6ch(df=df, ch0_present=ch0_present)
    df = add_variables.add_min_times(df=df)
    
    for func in [get_variables.get_ch_pe, get_variables.get_norm_ch_pe]:
        df = add_variables.add_mult_variables_to_df(df=df, get_var=func)
        
    for func, extra in [(get_variables.get_ch_pe_ns, dict(min_time=0, max_time=100)),
                        (get_variables.get_tmax, {}),
                        (get_variables.get_tmean, {}),
                        (get_variables.get_tlast, {})]:
        df = add_variables.add_mult_variables_to_df(df=df, get_var=func, ADC_MHz=ADC_MHz, **extra)
        
    df = add_variables.add_single_variable_to_df(df=df, get_var=get_variables.get_var_pe)
    
    for func in [get_variables.get_r_charge_bar, get_variables.get_x_charge_bar, get_variables.get_y_charge_bar]:
        df = add_variables.add_single_variable_to_df(df=df, get_var=func, ADC_MHz=ADC_MHz, min_time=0, max_time=max_time)
        
    for func in [get_variables.get_peaktime_a2ch, get_variables.get_peak_a2ch, get_variables.get_peaktime_a6ch, get_variables.get_peak_a6ch]:
        df = add_variables.add_single_variable_to_df(df=df, get_var=func, ADC_MHz=ADC_MHz, ch0_present=ch0_present, min_pe=2)
            
    asymmetry_time_vars = [
        get_variables.get_Asimmetry_6ch,
        get_variables.get_Asimmetry_4ch,
        get_variables.get_Asimmetry_2ch,
        get_variables.get_Asimmetry_dynamic,
        get_variables.get_Asimmetry_rms_2ch,
        get_variables.get_Asimmetry_rms_6ch,
        get_variables.get_Asimmetry_rms_dynamic,
        get_variables.get_Asimmetry_skew_6ch,
        get_variables.get_Asimmetry_hybmax_6ch,
        get_variables.get_Asimmetry_hyblast_6ch,
        get_variables.get_Asimmetry_hybmax_2ch,
        get_variables.get_Asimmetry_hyblast_2ch,
        get_variables.get_Asimmetry_q50,
        get_variables.get_Asymmetry_rise,
        get_variables.get_Asimmetry_60_240
    ]
    
    asymmetry_no_time_vars = [
        get_variables.get_Asimmetry_maxoplateau_6ch,
        get_variables.get_Asimmetry_maxoplateau_2ch
    ]
    
    for func in asymmetry_time_vars:
        df = add_variables.add_single_variable_to_df(
            df=df,
            get_var=func,
            ADC_MHz=ADC_MHz,
            min_time=0,
            max_time=max_time,
            ch0_present=ch0_present
        )
    
    for func in asymmetry_no_time_vars:
        df = add_variables.add_single_variable_to_df(
            df=df,
            get_var=func,
            ADC_MHz=ADC_MHz,
            ch0_present=ch0_present
        )
        
    df = df[(df["IsThereMuon"] == 0) | ((df["IsThereMuon"] == 1) & (df["frac_mu"] >= mu_frac))]

    table_df_frac_mu_undropped = df.copy()
    
    df_frac_mu_dropped = df.drop("frac_mu", axis = 1) 
    table_df_frac_mu_dropped = pa.Table.from_pandas(df_frac_mu_dropped)  #saves df without fraction of muon pe
    pq.write_table(table_df_frac_mu_dropped, save_path, compression="zstd")
    
    #df.to_csv(save_path, index = False)
    
    return table_df_frac_mu_undropped #WARNING : returns the fraction of muon pe which is forbidden for ML

def build_df_feats_8inches_withADC(trace_path : str, save_path : str, mu_frac : float = 0, pe_threshold = 1, pe_lim = np.inf, ADC_MHz = 1000):
    
    df = initialize_df_8inches.get_df(path = trace_path, base = "pe", ADC_MHz = ADC_MHz, pe_threshold = pe_threshold, pe_lim = pe_lim)
    df = add_variables_8inches.add_single_variable_to_df(df, get_var = get_variables_8inches.get_tmax, ADC_MHz = ADC_MHz)
    df = add_variables_8inches.add_single_variable_to_df(df, get_var = get_variables_8inches.get_tmean, ADC_MHz = ADC_MHz)
    df = add_variables_8inches.add_single_variable_to_df(df, get_var = get_variables_8inches.get_tlast, ADC_MHz = ADC_MHz)
    df = add_variables_8inches.add_single_variable_to_df(df, get_var = get_variables_8inches.get_ch_pe_ns, ADC_MHz = ADC_MHz, min_time = 0, max_time = 10)

    df = df.drop("ch0", axis = 1)
    df = df.drop("frac_mu", axis = 1)    
    df.to_csv(save_path, index = False)
    return 0

def build_df_feats_8inches_NOADC(trace_path : str, save_path : str, mu_frac : float = 0, pe_threshold = 1, pe_lim = np.inf, ch0_present = False, max_time : int = 100):
    
    ADC_MHz = None
    df = initialize_df.get_df(path = trace_path, base = "pe", ADC_MHz = ADC_MHz, pe_threshold = pe_threshold, pe_lim = pe_lim)
    df = add_variables.add_min_times(df)
    
    df = df.drop("ch_0", axis = 1)
    df = df.drop("ch_ref", axis = 1)
    df = df.drop("ch_60", axis = 1)
    df = df.drop("ch_120", axis = 1)
    df = df.drop("ch_180", axis = 1)
    df = df.drop("ch_240", axis = 1)
    df = df.drop("ch_300", axis = 1)

    df = df.drop("t_2min", axis = 1)
    df = df.drop("t_3min", axis = 1)
    df = df.drop("t_4min", axis = 1)
    df = df.drop("t_5min", axis = 1)
    df = df.drop("t_6min", axis = 1)
    df = df.drop("t_7min", axis = 1)

    df = df.drop("t_ref180", axis = 1)
    df = df.drop("t_ref120", axis = 1)
    df = df.drop("t_ref240", axis = 1)
    df = df.drop("t_ref300", axis = 1)
    df = df.drop("t_ref60", axis = 1)
    df = df.drop("t_ref0", axis = 1)
    df = df.drop("multiplicity", axis = 1)
    df = df.drop("t180", axis = 1)
    df = df.drop("tref", axis = 1)
    df = df.drop("t60", axis = 1)
    df = df.drop("t0", axis = 1)
    df = df.drop("t120", axis = 1)
    df = df.drop("t240", axis = 1)
    df = df.drop("t300", axis = 1)


    #usually useless, only keeping T_C : distance relative to core
    df = df.drop("Nom_Th", axis = 1)
    df = df.drop("Nom_En", axis = 1) 
    df = df.drop("Nom_Y_C", axis = 1)
    df = df.drop("Nom_X_C", axis = 1)
    df = df.drop("R_T", axis = 1)
    df = df.drop("X_T", axis = 1)
    df = df.drop("Y_T", axis = 1)

 
    df = df[(df["IsThereMuon"] == 0) | ((df["IsThereMuon"] == 1) & (df["frac_mu"] >= mu_frac))]

    

    table_df_frac_mu_undropped = df.copy()
    
    df_frac_mu_dropped = df.drop("frac_mu", axis = 1) 
    table_df_frac_mu_dropped = pa.Table.from_pandas(df_frac_mu_dropped)  #saves df without fraction of muon pe
    pq.write_table(table_df_frac_mu_dropped, save_path, compression="zstd")
    
    #df.to_csv(save_path, index = False)
    
    return table_df_frac_mu_undropped #WARNING : returns the fraction of muon pe which is forbidden for ML