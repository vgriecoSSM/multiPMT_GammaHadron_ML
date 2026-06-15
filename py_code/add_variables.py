import pandas as pd
import numpy as np 
from py_code import get_variables

def add_mult_variables_to_df(df: pd.DataFrame, get_var=None, *args, **kwargs):
    
    name_list = get_variables.get_name_var()[get_var]

    values = []
    trace_cols = ["ch_0", "ch_ref", "ch_60", "ch_120", "ch_180", "ch_240", "ch_300"]

    for row in df[trace_cols].itertuples(index=False, name=None):
        channels = list(row)
        values.append(get_var(channels, **kwargs))

    values = np.asarray(values)

    for j, name in enumerate(name_list):
        df[name] = values[:, j]

    return df

def add_single_variable_to_df(df: pd.DataFrame, get_var=None, *args, **kwargs):
    
    name_var = get_variables.get_name_var()[get_var]

    values = []
    trace_cols = ["ch_0", "ch_ref", "ch_60", "ch_120", "ch_180", "ch_240", "ch_300"]

    for row in df[trace_cols].itertuples(index=False, name=None):
        channels = list(row)
        values.append(get_var(channels, **kwargs))

    df[name_var] = values
    return df
    
def add_time_Asimmetry_2ch(df : pd.DataFrame, ch0_present = False):
    
    t0 = df["t0"]
    tref = df["tref"]
    t60 = df["t60"]
    t120 = df["t120"]
    t180 = df["t180"]
    t240 = df["t240"] 
    t300 = df["t300"]
    a_times = []  
    for i in range(df.shape[0]):
        front_times = np.nan
        if ch0_present == False : front_times = tref.iloc[i]
        else : front_times = tref.iloc[i] + t0.iloc[i]
        back_times = t180.iloc[i]
        if (front_times + back_times) != 0 : a_times.append((front_times - back_times)/(front_times + back_times))
        else : a_times.append(np.nan)
    df["a_time_2ch"] = a_times
    return df
          
def add_time_prompt_3ch(df : pd.DataFrame, ch0_present = False):

    t0 = df["t0"]
    tref = df["tref"]
    t60 = df["t60"]
    t120 = df["t120"]
    t180 = df["t180"]
    t240 = df["t240"] 
    t300 = df["t300"]
    cum_times = []  
    for i in range(df.shape[0]):
        front_times = np.nan
        if ch0_present == False : front_times = tref.iloc[i] + t60.iloc[i] + t300.iloc[i]
        else : front_times = t0.iloc[i] + tref.iloc[i] + t60.iloc[i] + t300.iloc[i]
        if front_times != 0 : cum_times.append(front_times)
        else : cum_times.append(np.nan)
    df["time_prompt_3ch"] = cum_times
    return df

def add_time_Asimmetry_6ch(df : pd.DataFrame, ch0_present = False):

    t0 = df["t0"]
    tref = df["tref"]
    t60 = df["t60"]
    t120 = df["t120"]
    t180 = df["t180"]
    t240 = df["t240"] 
    t300 = df["t300"]
    a_times = []  
    for i in range(df.shape[0]):
        front_times = np.nan
        if ch0_present == False : front_times = tref.iloc[i] + t60.iloc[i] + t300.iloc[i]
        else : front_times = tref.iloc[i] + t60.iloc[i] + t300.iloc[i] + t0.iloc[i]
        back_times = t180.iloc[i] + t120.iloc[i] + t240.iloc[i]
        if (front_times + back_times) != 0 : a_times.append((front_times - back_times)/(front_times + back_times))
        else : a_times.append(np.nan)
    df["a_time_6ch"] = a_times
    return df

def add_multiplicity(df = pd.DataFrame, pe_trigger = 2, time_frame = 10):

    ch0 = df["ch_0"]
    chref = df["ch_ref"]
    ch60 = df["ch_60"]
    ch120 = df["ch_120"]
    ch180 = df["ch_180"]
    ch240 = df["ch_240"]
    ch300 = df["ch_300"]
    multiplicity = []
    for i in range(df.shape[0]):
        channels = [ch0.iloc[i], chref.iloc[i], ch60.iloc[i], ch120.iloc[i], ch180.iloc[i], ch240.iloc[i], ch300.iloc[i]]
        multiplicity_val = get_variables.get_multiplicity(channels, pe_trigger, time_frame)
        multiplicity.append(multiplicity_val)
    df["multiplicity"] = multiplicity
    return df

def add_rel_time(df = pd.DataFrame):
    t0 = np.array(df["t0"])
    tref = np.array(df["tref"])
    t60 = np.array(df["t60"])
    t120 = np.array(df["t120"])
    t180 = np.array(df["t180"])
    t240 = np.array(df["t240"])
    t300 = np.array(df["t300"])
    ref0 = t0 - tref
    ref60 = t60 - tref
    ref120 = t120 - tref
    ref180 = t180 - tref
    ref240 = t240 - tref
    ref300 = t300 - tref
    df["t_ref0"] = ref0
    df["t_ref60"] = ref60
    df["t_ref120"] = ref120
    df["t_ref180"] = ref180
    df["t_ref240"] = ref240
    df["t_ref300"] = ref300
    return df

def add_times_rel_to_min_time(df : pd.DataFrame):
    t0 = df["t0"]
    tref = df["tref"]
    t60 = df["t60"]
    t120 = df["t120"]
    t180 = df["t180"]
    t240 = df["t240"] 
    t300 = df["t300"]
    event_times = np.array([[t0.iloc[i], tref.iloc[i], t60.iloc[i], t120.iloc[i], t180.iloc[i], t240.iloc[i], t300.iloc[i]] for i in range(len(t0))])
    min_event_times = np.array([np.nanmin(event_times[i]) for i in range(len(event_times))])
    
    t0_rel_min = round(t0 - min_event_times, 1)
    tref_rel_min = round(tref - min_event_times,1)
    t60_rel_min = round(t60 - min_event_times,1)
    t120_rel_min = round(t120 - min_event_times,1)
    t180_rel_min = round(t180 - min_event_times,1)
    t240_rel_min = round(t240 - min_event_times,1)
    t300_rel_min = round(t300 - min_event_times,1)
    

    df["t0_rel_min"] = t0_rel_min
    df["tref_rel_min"] = tref_rel_min
    df["t60_rel_min"] = t60_rel_min
    df["t120_rel_min"] = t120_rel_min
    df["t180_rel_min"] = t180_rel_min
    df["t240_rel_min"] = t240_rel_min
    df["t300_rel_min"] = t300_rel_min

    return df_with_time_rel

def add_min_times(df : pd.DataFrame):
    
    t0 = df["t0"]
    tref = df["tref"]
    t60 = df["t60"]
    t120 = df["t120"]
    t180 = df["t180"]
    t240 = df["t240"] 
    t300 = df["t300"]
    
    event_times = np.array([[t0.iloc[i], tref.iloc[i], t60.iloc[i], t120.iloc[i], t180.iloc[i], t240.iloc[i], t300.iloc[i]] for i in range(len(t0))])
    min_event_times = np.array([np.sort(event_times[i]) for i in range(len(event_times))])
    
    t_min = [min_event_times[i][0] for i in range(len(min_event_times))]
    t_2min = [min_event_times[i][1] for i in range(len(min_event_times))]
    t_3min = [min_event_times[i][2] for i in range(len(min_event_times))]
    t_4min = [min_event_times[i][3] for i in range(len(min_event_times))]
    t_5min = [min_event_times[i][4] for i in range(len(min_event_times))]
    t_6min = [min_event_times[i][5] for i in range(len(min_event_times))]
    t_7min = [min_event_times[i][6] for i in range(len(min_event_times))]
    
    df["t_min"] = t_min
    df["t_2min"] = t_2min
    df["t_3min"] = t_3min
    df["t_4min"] = t_4min
    df["t_5min"] = t_5min
    df["t_6min"] = t_6min
    df["t_7min"] = t_7min

    return df