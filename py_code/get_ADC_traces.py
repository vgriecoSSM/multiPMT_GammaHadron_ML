import numpy as np
import pandas as pd
from py_code import get_ADC_traces

def get_ADC_sampled_df(df, ADC_MHz : int | None = 250, input_time_resolution : float = 1.0):


    if ADC_MHz is None:

        df_samp = df.copy()

        df_samp["ch_0"] = df["ch_0"].apply(lambda x: [sum(x)])
        df_samp["ch_ref"] = df["ch_ref"].apply(lambda x: [sum(x)])
        df_samp["ch_60"] = df["ch_60"].apply(lambda x: [sum(x)])
        df_samp["ch_120"] = df["ch_120"].apply(lambda x: [sum(x)])
        df_samp["ch_180"] = df["ch_180"].apply(lambda x: [sum(x)])
        df_samp["ch_240"] = df["ch_240"].apply(lambda x: [sum(x)])
        df_samp["ch_300"] = df["ch_300"].apply(lambda x: [sum(x)])

        return df_samp

    
    adc_time_resolution = 1000 / ADC_MHz

    samples_per_adc_bin = int(round(adc_time_resolution/ input_time_resolution))
                             
    ch0 = df["ch_0"]
    chref = df["ch_ref"]
    ch60 = df["ch_60"]
    ch120 = df["ch_120"]
    ch180 = df["ch_180"]
    ch240 = df["ch_240"]
    ch300 = df["ch_300"]
    ch0_samp = []
    chref_samp = []
    ch60_samp = []
    ch120_samp = []
    ch180_samp = []
    ch240_samp = []
    ch300_samp = []
    
    for i in range(df.shape[0]):
            ch0_evt = [sum(ch0.iloc[i][j:j+samples_per_adc_bin]) for j in range(0, len(ch0.iloc[i]), samples_per_adc_bin)]
            ch0_samp.append(ch0_evt)
            chref_evt = [sum(chref.iloc[i][j:j+samples_per_adc_bin]) for j in range(0, len(chref.iloc[i]), samples_per_adc_bin)]
            chref_samp.append(chref_evt)
            ch60_evt = [sum(ch60.iloc[i][j:j+samples_per_adc_bin]) for j in range(0, len(ch60.iloc[i]), samples_per_adc_bin)]
            ch60_samp.append(ch60_evt) 
            ch120_evt = [sum(ch120.iloc[i][j:j+samples_per_adc_bin]) for j in range(0, len(ch120.iloc[i]), samples_per_adc_bin)]
            ch120_samp.append(ch120_evt)
            ch180_evt = [sum(ch180.iloc[i][j:j+samples_per_adc_bin]) for j in range(0, len(ch180.iloc[i]), samples_per_adc_bin)]
            ch180_samp.append(ch180_evt)
            ch240_evt = [sum(ch240.iloc[i][j:j+samples_per_adc_bin]) for j in range(0, len(ch240.iloc[i]), samples_per_adc_bin)]
            ch240_samp.append(ch240_evt)
            ch300_evt = [sum(ch300.iloc[i][j:j+samples_per_adc_bin]) for j in range(0, len(ch300.iloc[i]), samples_per_adc_bin)]
            ch300_samp.append(ch300_evt)
        
    df_samp = df.copy()
    df_samp["ch_0"] = ch0_samp
    df_samp["ch_ref"] = chref_samp
    df_samp["ch_60"] = ch60_samp
    df_samp["ch_120"] = ch120_samp
    df_samp["ch_180"] = ch180_samp
    df_samp["ch_240"] = ch240_samp
    df_samp["ch_300"] = ch300_samp
    return df_samp            