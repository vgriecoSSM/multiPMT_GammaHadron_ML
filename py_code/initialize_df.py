import numpy as np
import pandas as pd
import json
from py_code import get_ADC_traces
from py_code import add_variables
from py_code import get_variables

def get_fpe_time_ch_indx(channels :list):
    times = [np.nan for i in range(6)]
    cum_pe = [0 for i in range(6)]
    for ch_indx, ch in enumerate(channels): 
        val = 0
        for indx, pe in enumerate(ch):
            val += pe
            if val >= 1:
                times[ch_indx] = indx
                cum_pe[ch_indx] = val
                break
    if not np.all(np.isnan(times)):
        min_time = np.nanmin(times)
        idx_candidates = [i for i in range(len(times)) if times[i] == min_time]
        hi_pe = max([cum_pe[i] for i in idx_candidates])
        hi_pe_indx = idx_candidates[[cum_pe[i] for i in idx_candidates].index(hi_pe)]
        return hi_pe_indx
    else:
        return np.nan

def get_max_ch_indx(channels: list):
    
        ch1_pe = sum(channels[0])
        ch2_pe = sum(channels[1])
        ch3_pe = sum(channels[2])
        ch4_pe = sum(channels[3])
        ch5_pe = sum(channels[4])
        ch6_pe = sum(channels[5])
        pe_list = [ch1_pe, ch2_pe, ch3_pe, ch4_pe, ch5_pe, ch6_pe]
        if pe_list:
            max_val = max(pe_list)
            idx_candidates = [i for i in range(len(pe_list)) if pe_list[i] == max_val]
            max_indx = np.random.choice(idx_candidates)
            return max_indx
        else:
            return np.nan

def traces_to_lists(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: json.loads(str(x)) if isinstance(x, str) or isinstance(x, (bytes, bytearray)) else x)  
    return df
    
def get_mu_df(df : pd.DataFrame):
    df = df[df["IsThereMuon"]==1].copy()
    return df
    
def get_nomu_df(df : pd.DataFrame):
    df = df[df["IsThereMuon"]==0].copy()
    return df   

def get_ord_df(df : pd.DataFrame, base : str):
        ch_ref = []
        ch_60 = []
        ch_120 = []
        ch_180 = []
        ch_240 = []
        ch_300 = []
        ch1 = df["ch1"]
        ch2 = df["ch2"]
        ch3 = df["ch3"]
        ch4 = df["ch4"]
        ch5 = df["ch5"]
        ch6 = df["ch6"]
        for i in range(df.shape[0]):
            channels = [ch1.iloc[i], ch2.iloc[i], ch3.iloc[i], ch4.iloc[i], ch5.iloc[i], ch6.iloc[i]]
            idx_ref = np.nan
            if base == "time":
                idx_ref = get_fpe_time_ch_indx(channels)
            elif base == "pe":
                idx_ref = get_max_ch_indx(channels)   
            if not np.isnan(idx_ref):
                for j in range(len(channels)):
                        if j == idx_ref : 
                            ch_ref.append(channels[idx_ref])
                            ch_60.append(channels[(idx_ref - 5) % 6])
                            ch_120.append(channels[(idx_ref - 4) % 6])
                            ch_180.append(channels[(idx_ref - 3) % 6])
                            ch_240.append(channels[(idx_ref - 2) % 6])
                            ch_300.append(channels[(idx_ref - 1) % 6])
                            break
            else : 
                ch_ref.append(channels[0])
                ch_300.append(channels[5])
                ch_240.append(channels[4])
                ch_180.append(channels[3])
                ch_120.append(channels[2])
                ch_60.append(channels[1])
                
        ordered_df = pd.DataFrame({
            "ch_0": df["ch0"],
            "ch_ref": ch_ref,
            "ch_60": ch_60,
            "ch_120": ch_120,
            "ch_180": ch_180,
            "ch_240": ch_240,
            "ch_300": ch_300,
            **({"X_T": df["PE_XTank"]} if "PE_XTank" in df.columns else {}),
            **({"Y_T": df["PE_YTank"]} if "PE_YTank" in df.columns else {}),
            **({"R_T": df["PE_RTank"]} if "PE_RTank" in df.columns else {}),
            **({"Nom_En": df["Nominal_Energy"]} if "Nominal_Energy" in df.columns else {}),
            **({"Nom_Th": df["Nominal_Theta"]} if "Nominal_Theta" in df.columns else {}),
            **({"Nom_X_C": df["Nominal_XCore"]} if "Nominal_XCore" in df.columns else {}),
            **({"Nom_Y_C": df["Nominal_YCore"]} if "Nominal_YCore" in df.columns else {}),
            **({"T_C": df["T_C"]} if "T_C" in df.columns else {}),
            **({"frac_mu": df["frac_mu"]} if "frac_mu" in df.columns else {}),
            "IsThereMuon": df["IsThereMuon"]
            })
        return ordered_df

def get_pefrac_mu(df: pd.DataFrame):

    smu_cols = [f"ch{i}_smu" for i in range(7)]
    nmu_cols = [f"ch{i}_nmu" for i in range(7)]
    required_cols = smu_cols + nmu_cols

    if not all(c in df.columns for c in required_cols):
        return df

    df = df.copy()

    frac_pe_mu = []
    for i in range(df.shape[0]):
        pe_smu = 0
        pe_nmu = 0
        frac = 0
        channels_smu = [df[c].iloc[i] for c in smu_cols]
        channels_nmu = [df[c].iloc[i] for c in nmu_cols]
        pe_smu = np.sum(np.asarray(channels_smu))
        pe_nmu = np.sum(np.asarray(channels_nmu))
        if pe_smu + pe_nmu > 0:
            frac = pe_smu / (pe_smu + pe_nmu)
        frac_pe_mu.append(frac)

    df["frac_mu"] = frac_pe_mu

    return df

    
def get_df(path : str, base : str, ADC_MHz : int, pe_threshold : int, pe_lim : int = np.inf):
    #df_raw = pd.read_csv(path, sep = " ") 
    df_list = pd.read_parquet(path)
    #df_list = traces_to_lists(df_raw)  
    df_wfrac = get_pefrac_mu(df_list)
    df_ord = get_ord_df(df_wfrac, base  = base)
    df_ord_with_times = add_variables.add_mult_variables_to_df(df_ord, get_variables.get_first_time) 
    df_ord_with_rel_times = add_variables.add_rel_time(df_ord_with_times) 
    df_with_mult = add_variables.add_multiplicity(df_ord_with_rel_times, pe_trigger = 2, time_frame = 10)
    df = get_ADC_traces.get_ADC_sampled_df(df_with_mult, ADC_MHz = ADC_MHz)
    df = add_variables.add_single_variable_to_df(df = df, get_var = get_variables.get_total_pe)
    df = df[df["total_pe"]>= pe_threshold]
    df = df[df["total_pe"]<= pe_lim]
    return df