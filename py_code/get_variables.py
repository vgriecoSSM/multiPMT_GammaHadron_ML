import numpy as np
import pandas as pd
import math
from scipy.stats import skew

def get_name_var():
    var_dict = {get_first_time : ["t0", "tref","t60","t120","t180","t240","t300"],
                get_total_pe : "total_pe",
                get_Asimmetry_6ch : "a_6ch",
                get_Asimmetry_4ch : "a_4ch",
                get_Asimmetry_2ch : "a_2ch",
                get_multiplicity : "multiplicity",
                get_ch_pe : ["ch0_pe", "chref_pe", "ch60_pe", "ch120_pe", "ch180_pe", "ch240_pe", "ch300_pe"],
                get_norm_ch_pe : ["norm_ch0_pe", "norm_chref_pe", "norm_ch60_pe", "norm_ch120_pe", "norm_ch180_pe", "norm_ch240_pe", "norm_ch300_pe"],
                get_Asimmetry_dynamic : "a_dyn",
                get_Asimmetry_rms_2ch : "a_rms_2ch",
                get_Asimmetry_rms_6ch : "a_rms_6ch",
                get_Asimmetry_rms_dynamic : "a_rms_dyn",
                get_Asimmetry_skew_6ch : "a_skew_6ch",
                get_peaktime_a2ch : "peaktime_a2ch",
                get_peak_a2ch : "peak_a2ch",
                get_peaktime_a6ch : "peaktime_a6ch",
                get_peak_a6ch : "peak_a6ch",
                get_r_charge_bar : "r_bar",
                get_x_charge_bar : "x_bar",
                get_y_charge_bar : "y_bar",
                get_tmax : ["t0_max", "tref_max","t60_max","t120_max","t180_max","t240_max","t300_max"],
                get_tmean : ["t0_mean", "tref_mean","t60_mean","t120_mean","t180_mean","t240_mean","t300_mean"],
                get_tlast : ["t0_last", "tref_last","t60_last","t120_last","t180_last","t240_last","t300_last"],
                get_Asimmetry_hybmax_6ch : "a_hybmax_6ch",
                get_Asimmetry_hyblast_6ch : "a_hyblast_6ch",
                get_Asimmetry_hybmax_2ch : "a_hybmax_2ch",
                get_Asimmetry_hyblast_2ch : "a_hyblast_2ch",
                get_Asimmetry_maxoplateau_6ch : "a_maxoplateau_6ch",
                get_Asimmetry_maxoplateau_2ch : "a_maxoplateau_2ch",
                get_Asimmetry_q50 : "a_q50",
                get_Asymmetry_rise : "a_rise",
                get_Asimmetry_60_240 : "a_60_240",
                get_var_pe : "var",
                get_ch_pe_ns : ["ch0_ns", "chref_ns", "ch60_ns", "ch120_ns", "ch180_ns", "ch240_ns", "ch300_ns"],
                get_t_rise_10_50 : ["t_rise_10_50_ch0", "t_rise_10_50_chref", "t_rise_10_50_ch60", "t_rise_10_50_ch120", "t_rise_10_50_ch180", "t_rise_10_50_ch240", "t_rise_10_50_ch300"],
                get_t_rise_50_90 : ["t_rise_50_90_ch0", "t_rise_50_90_chref", "t_rise_50_90_ch60", "t_rise_50_90_ch120", "t_rise_50_90_ch180", "t_rise_50_90_ch240", "t_rise_50_90_ch300"],
                get_t_rise_10_90 : ["t_rise_10_90_ch0", "t_rise_10_90_chref", "t_rise_10_90_ch60", "t_rise_10_90_ch120", "t_rise_10_90_ch180", "t_rise_10_90_ch240", "t_rise_10_90_ch300"],
                get_Asimmetry_6ch_foractivechannels : "a_6ch_activechs"
               }
    return var_dict

def get_first_time(channels : list, pe_thresh : int = 1): 
    times = [np.nan for i in range(7)]
    for ch_indx, ch in enumerate(channels): 
        val = 0
        for indx, pe in enumerate(ch):
            val += pe
            if val>= pe_thresh:
                times[ch_indx] = indx
                break
    times = [i*0.2 for i in times]
    return times

def get_total_pe(channels : list):
    ch0 = sum(channels[0])
    chref = sum(channels[1])
    ch60 = sum(channels[2])
    ch120 = sum(channels[3])
    ch180 = sum(channels[4])
    ch240 = sum(channels[5])
    ch300 = sum(channels[6])
    tot_pe = ch0 + chref + ch60 + ch120 + ch180 + ch240 + ch300
    return tot_pe

def get_Asimmetry_6ch(channels : list, min_time = 0 , max_time = 100, ch0_present = False, ADC_MHz = 1000):

    if ADC_MHz is None:
        
        ch0 = sum(channels[0])
        chref = sum(channels[1])
        ch60 = sum(channels[2])
        ch120 = sum(channels[3])
        ch180 = sum(channels[4])
        ch240 = sum(channels[5])
        ch300 = sum(channels[6])

    else:
        
        t_samp = 1000/ADC_MHz
        max_indx = math.ceil(max_time/t_samp)
        min_indx = math.floor(min_time/t_samp)
        ch0 = sum(channels[0][min_indx:max_indx])
        chref = sum(channels[1][min_indx:max_indx])
        ch60 = sum(channels[2][min_indx:max_indx])
        ch120 = sum(channels[3][min_indx:max_indx])
        ch180 = sum(channels[4][min_indx:max_indx])
        ch240 = sum(channels[5][min_indx:max_indx])
        ch300 = sum(channels[6][min_indx:max_indx])
        
    front_ch = np.nan
    if ch0_present == False : front_ch = chref + ch60 + ch300
    else : front_ch = chref + ch60 + ch300 +ch0
    back_ch = ch120 + ch180 + ch240
    if (front_ch + back_ch) != 0 : return (front_ch - back_ch)/(front_ch + back_ch)
    else : return np.nan
        
def get_Asimmetry_4ch(channels : list, min_time = 0 , max_time = 100, ch0_present = False, ADC_MHz = 1000):

    if ADC_MHz is None:
        
        ch0 = sum(channels[0])
        chref = sum(channels[1])
        ch60 = sum(channels[2])
        ch120 = sum(channels[3])
        ch180 = sum(channels[4])
        ch240 = sum(channels[5])
        ch300 = sum(channels[6])

    else:
            
        t_samp = 1000/ADC_MHz
        max_indx = math.ceil(max_time/t_samp)
        min_indx = math.floor(min_time/t_samp)
        ch0 = sum(channels[0][min_indx:max_indx])
        chref = sum(channels[1][min_indx:max_indx])
        ch60 = sum(channels[2][min_indx:max_indx])
        ch120 = sum(channels[3][min_indx:max_indx])
        ch180 = sum(channels[4][min_indx:max_indx])
        ch240 = sum(channels[5][min_indx:max_indx])
        ch300 = sum(channels[6][min_indx:max_indx])
        
    front_ch = np.nan
    if ch0_present == False : front_ch = max(chref + ch60, chref + ch300) 
    else : front_ch = max(chref + ch60 + ch0, chref + ch300 + ch0)
    back_ch = min(ch120 + ch180, ch180 + ch240)
    if (front_ch + back_ch) != 0 : return (front_ch - back_ch)/(front_ch + back_ch)
    else : return np.nan
        
def get_Asimmetry_2ch(channels : list, min_time = 0 , max_time = 100, ch0_present = False, ADC_MHz = 1000):

    if ADC_MHz is None:
    
        ch0 = sum(channels[0])
        chref = sum(channels[1])
        ch60 = sum(channels[2])
        ch120 = sum(channels[3])
        ch180 = sum(channels[4])
        ch240 = sum(channels[5])
        ch300 = sum(channels[6])

    else:
    
        t_samp = 1000/ADC_MHz
        max_indx = math.ceil(max_time/t_samp)
        min_indx = math.floor(min_time/t_samp)
        ch0 = sum(channels[0][min_indx:max_indx])
        chref = sum(channels[1][min_indx:max_indx])
        ch60 = sum(channels[2][min_indx:max_indx])
        ch120 = sum(channels[3][min_indx:max_indx])
        ch180 = sum(channels[4][min_indx:max_indx])
        ch240 = sum(channels[5][min_indx:max_indx])
        ch300 = sum(channels[6][min_indx:max_indx])
        
    front_ch = np.nan
    if ch0_present == False : front_ch = chref 
    else : front_ch = chref + ch0
    back_ch = ch180
    if (front_ch + back_ch) != 0 : return (front_ch - back_ch)/(front_ch + back_ch)
    else : return np.nan
        
def get_inst_Asimmetry_2ch(channels : list, ch0_present = False, min_pe = 2):
    ch0 = channels[0]
    chref = channels[1]
    ch60 = channels[2]
    ch120 = channels[3]
    ch180 = channels[4]
    ch240 = channels[5]
    ch300 = channels[6]
    front_ch = np.nan
    if ch0_present == False : front_ch = chref 
    else : front_ch = chref + ch0
    back_ch = ch180 
    if (front_ch + back_ch) != 0 and front_ch >= min_pe : return (front_ch - back_ch)/(front_ch + back_ch)
    else : return np.nan

def get_inst_Asimmetry_6ch(channels : list, ch0_present = False, min_pe = 2):
    ch0 = channels[0]
    chref = channels[1]
    ch60 = channels[2]
    ch120 = channels[3]
    ch180 = channels[4]
    ch240 = channels[5]
    ch300 = channels[6]
    front_ch = np.nan
    if ch0_present == False : front_ch = chref + ch60 + ch300
    else : front_ch = chref + ch0 + ch60 + ch300
    back_ch = ch180 + ch120 + ch240
    if (front_ch + back_ch) != 0 and front_ch >= min_pe : return (front_ch - back_ch)/(front_ch + back_ch)
    else : return np.nan

def get_multiplicity(channels : list, pe_trigger = 2, time_frame = 10):
    times = get_first_time(channels, pe_thresh = pe_trigger)
    if np.any(~np.isnan(times)):
        first_trigger_time = np.nanmin(times)
        open_window = first_trigger_time + time_frame
        
        t_samp = 0.2
        ftt_indx = math.ceil(first_trigger_time/t_samp)
        opw_indx = math.ceil(open_window/t_samp)
        
        ch0_w = sum(channels[0][ftt_indx:opw_indx])
        chref_w = sum(channels[1][ftt_indx:opw_indx])
        ch60_w = sum(channels[2][ftt_indx:opw_indx])
        ch120_w = sum(channels[3][ftt_indx:opw_indx])
        ch180_w = sum(channels[4][ftt_indx:opw_indx])
        ch240_w = sum(channels[5][ftt_indx:opw_indx])
        ch300_w = sum(channels[6][ftt_indx:opw_indx])
    
        w_ch = np.array([ch0_w, chref_w, ch60_w, ch120_w, ch180_w, ch240_w, ch300_w])
        multiplicity = np.sum(w_ch >= pe_trigger)
        if multiplicity!= 0 : return multiplicity
        else : return np.nan
    else : return np.nan    

def get_ch_pe(channels : list):
    ch0 = sum(channels[0])
    chref = sum(channels[1])
    ch60 = sum(channels[2])
    ch120 = sum(channels[3])
    ch180 = sum(channels[4])
    ch240 = sum(channels[5])
    ch300 = sum(channels[6])
    chs = [ch0, chref, ch60, ch120, ch180, ch240, ch300]
    return chs

def get_var_pe(channels : list):
    ch0 = sum(channels[0])
    chref = sum(channels[1])
    ch60 = sum(channels[2])
    ch120 = sum(channels[3])
    ch180 = sum(channels[4])
    ch240 = sum(channels[5])
    ch300 = sum(channels[6])
    chs = np.array([ch0, chref, ch60, ch120, ch180, ch240, ch300])
    var_chs = np.var(chs)
    return var_chs

def get_norm_ch_pe(channels : list):
    tot_pe = get_total_pe(channels)
    chs = get_ch_pe(channels)
    ch_norm = [i/tot_pe if tot_pe != 0 else 0 for i in chs]
    return ch_norm

def get_ch_pe_ns(channels : list, min_time = 0 , max_time = 20, ADC_MHz = 5000):
    
    t_samp = 1000/ADC_MHz
    max_indx = math.ceil(max_time/t_samp)
    min_indx = math.floor(min_time/t_samp)
    ch0_ns = sum(channels[0][min_indx:max_indx])
    chref_ns = sum(channels[1][min_indx:max_indx])
    ch60_ns = sum(channels[2][min_indx:max_indx])
    ch120_ns = sum(channels[3][min_indx:max_indx])
    ch180_ns = sum(channels[4][min_indx:max_indx])
    ch240_ns = sum(channels[5][min_indx:max_indx])
    ch300_ns = sum(channels[6][min_indx:max_indx])  
    
    chs_ns = [ch0_ns, chref_ns, ch60_ns, ch120_ns, ch180_ns, ch240_ns, ch300_ns]
    return chs_ns

def get_Asimmetry_dynamic(channels : list, min_time = 0 , max_time = 100, ch0_present = False, ADC_MHz = 1000):

    if ADC_MHz is None:
    
        ch0 = sum(channels[0])
        chref = sum(channels[1])
        ch60 = sum(channels[2])
        ch120 = sum(channels[3])
        ch180 = sum(channels[4])
        ch240 = sum(channels[5])
        ch300 = sum(channels[6])

    else:
    
        t_samp = 1000/ADC_MHz
        max_indx = math.ceil(max_time/t_samp)
        min_indx = math.floor(min_time/t_samp)
        ch0 = sum(channels[0][min_indx:max_indx])
        chref = sum(channels[1][min_indx:max_indx])
        ch60 = sum(channels[2][min_indx:max_indx])
        ch120 = sum(channels[3][min_indx:max_indx])
        ch180 = sum(channels[4][min_indx:max_indx])
        ch240 = sum(channels[5][min_indx:max_indx])
        ch300 = sum(channels[6][min_indx:max_indx])   
        
    if ch60 >= ch300 :  
        near_ch = ch60 
        opposite_ch = ch240
    else : 
        near_ch = ch300
        opposite_ch = ch120
    if ch0_present == False : front_ch = chref + near_ch
    else : front_ch = chref + near_ch + ch0
    back_ch = ch180 + opposite_ch
    if (front_ch + back_ch) != 0 : return (front_ch - back_ch)/(front_ch + back_ch)
    else : return np.nan

def get_Asimmetry_rms_2ch(channels : list, min_time=0, max_time=100, ch0_present=False, ADC_MHz=1000):
    t_samp = 1000/ADC_MHz
    max_indx = math.ceil(max_time/t_samp)
    min_indx = math.floor(min_time/t_samp)
    ch_tr = [np.array(ch[min_indx:max_indx]) for ch in channels]
    ch0 = ch_tr[0]  
    chref = ch_tr[1]
    ch60  = ch_tr[2]
    ch120 = ch_tr[3]
    ch180 = ch_tr[4]
    ch240 = ch_tr[5]
    ch300 = ch_tr[6]
    if ch0_present == False : front_inst = chref
    else : front_inst = chref + ch0
    back_inst = ch180
    if np.all(back_inst == 0) and np.all(front_inst == 0):
        num_inst = np.full_like(front_inst, np.nan)
        den_inst = np.full_like(front_inst, np.nan)
    else :
        num_inst = front_inst - back_inst
        den_inst = front_inst + back_inst 
    num_cum = np.cumsum(num_inst) 
    den_cum = np.cumsum(den_inst) 
    if np.any(den_cum != 0) : a_cum = num_cum/(den_cum + 1.e-9)
    else : a_cum = np.nan
    a_rms = float(np.sqrt(np.nanmean(a_cum**2)))
    return a_rms

def get_Asimmetry_rms_6ch(channels : list, min_time=0, max_time=100, ch0_present=False, ADC_MHz=1000):
    t_samp = 1000/ADC_MHz
    max_indx = math.ceil(max_time/t_samp)
    min_indx = math.floor(min_time/t_samp)
    ch_tr = [np.array(ch[min_indx:max_indx]) for ch in channels]
    ch0 = ch_tr[0]  
    chref = ch_tr[1]
    ch60  = ch_tr[2]
    ch120 = ch_tr[3]
    ch180 = ch_tr[4]
    ch240 = ch_tr[5]
    ch300 = ch_tr[6]
    if ch0_present == False : front_inst = chref + ch60 + ch300
    else : front_inst = chref + ch60 + ch300 + ch0
    back_inst = ch180 + ch120 + ch240
    if np.all(back_inst == 0) and np.all(front_inst == 0):
        num_inst = np.full_like(front_inst, np.nan)
        den_inst = np.full_like(front_inst, np.nan)
    else :
        num_inst = front_inst - back_inst
        den_inst = front_inst + back_inst 
    num_cum = np.cumsum(num_inst) 
    den_cum = np.cumsum(den_inst) 
    if np.any(den_cum != 0) : a_cum = num_cum/(den_cum + 1.e-9)
    else : a_cum = np.nan
    a_rms = float(np.sqrt(np.nanmean(a_cum**2)))
    return a_rms

def get_Asimmetry_rms_dynamic(channels : list, min_time=0, max_time=100, ch0_present=False, ADC_MHz=1000):
    t_samp = 1000/ADC_MHz
    max_indx = math.ceil(max_time/t_samp)
    min_indx = math.floor(min_time/t_samp)
    ch_tr = [np.array(ch[min_indx:max_indx]) for ch in channels]
    ch0 = ch_tr[0]  
    chref = ch_tr[1]
    ch60  = ch_tr[2]
    ch120 = ch_tr[3]
    ch180 = ch_tr[4]
    ch240 = ch_tr[5]
    ch300 = ch_tr[6]
    if np.sum(ch60) >= np.sum(ch300) :  
        near_inst = ch60 
        opposite_inst = ch240
    else : 
        near_inst = ch300
        opposite_inst = ch120
    if ch0_present == False : front_inst = chref + near_inst
    else : front_inst = chref + near_inst + ch0
    back_inst = ch180 + opposite_inst
    if np.all(back_inst == 0) and np.all(front_inst == 0):
        num_inst = np.full_like(front_inst, np.nan)
        den_inst = np.full_like(front_inst, np.nan)
    else :
        num_inst = front_inst - back_inst
        den_inst = front_inst + back_inst 
    num_cum = np.cumsum(num_inst) 
    den_cum = np.cumsum(den_inst) 
    if np.any(den_cum != 0) : a_cum = num_cum/(den_cum + 1.e-9)
    else : a_cum = np.nan
    a_rms = float(np.sqrt(np.nanmean(a_cum**2)))
    return a_rms

def get_Asimmetry_skew_6ch(channels : list, min_time=0, max_time=100, ch0_present=False, ADC_MHz=1000):
    t_samp = 1000/ADC_MHz
    max_indx = math.ceil(max_time/t_samp)
    min_indx = math.floor(min_time/t_samp)
    ch_tr = [np.array(ch[min_indx:max_indx]) for ch in channels]
    ch0 = ch_tr[0]
    chref = ch_tr[1]
    ch60  = ch_tr[2]
    ch120 = ch_tr[3]
    ch180 = ch_tr[4]
    ch240 = ch_tr[5]
    ch300 = ch_tr[6]
    if ch0_present == False : front_inst = chref + ch60 + ch300
    else : front_inst = chref + ch60 + ch300 + ch0
    back_inst = ch180 + ch120 + ch240
    secs = np.arange(front_inst.size) * t_samp
    if np.all(back_inst == 0) and np.all(front_inst == 0):
        s_pos = np.nan
        s_neg = np.nan
    else:
        s_pos = skew(front_inst * secs)
        s_neg = skew(back_inst * secs)
    if np.isnan(s_pos) or np.isnan(s_neg):
        a_skew = np.nan
    else:
        a_skew = (s_pos - s_neg)/(s_pos + s_neg + 1.e-9)
    return a_skew

def get_peaktime_a2ch(channels : list, ch0_present = False, ADC_MHz = 5000, min_pe = 2):
    t_samp = 1000/ADC_MHz
    ch0 = channels[0]
    chref = channels[1]
    ch60 = channels[2]
    ch120 = channels[3]
    ch180 = channels[4]
    ch240 = channels[5]
    ch300 = channels[6]
    inst_a2ch = [get_inst_Asimmetry_2ch(channels = [ch0[i], chref[i], ch60[i], ch120[i], ch180[i], ch240[i], ch300[i]], ch0_present = ch0_present, min_pe = min_pe) for i in range(len(ch0))]
    inst_a2ch = np.array(inst_a2ch)
    if np.all(np.isnan(inst_a2ch)) : peaktime_ns = np.nan
    else :
        max_indx = np.nanargmax(inst_a2ch)
        peaktime_ns = max_indx*t_samp
    return peaktime_ns

def get_peak_a2ch(channels : list, ch0_present = False, ADC_MHz = 5000, min_pe = 2):
    t_samp = 1000/ADC_MHz
    ch0 = channels[0]
    chref = channels[1]
    ch60 = channels[2]
    ch120 = channels[3]
    ch180 = channels[4]
    ch240 = channels[5]
    ch300 = channels[6]
    inst_a2ch = [get_inst_Asimmetry_2ch(channels = [ch0[i], chref[i], ch60[i], ch120[i], ch180[i], ch240[i], ch300[i]], ch0_present = ch0_present, min_pe = min_pe) for i in range(len(ch0))]
    inst_a2ch = np.array(inst_a2ch)
    if np.all(np.isnan(inst_a2ch)) : max_a2ch = np.nan
    else :
        max_a2ch = np.nanmax(inst_a2ch)
    return max_a2ch

def get_peak_a6ch(channels : list, ch0_present = False, ADC_MHz = 5000, min_pe = 2):
    t_samp = 1000/ADC_MHz
    ch0 = channels[0]
    chref = channels[1]
    ch60 = channels[2]
    ch120 = channels[3]
    ch180 = channels[4]
    ch240 = channels[5]
    ch300 = channels[6]
    inst_a6ch = [get_inst_Asimmetry_6ch(channels = [ch0[i], chref[i], ch60[i], ch120[i], ch180[i], ch240[i], ch300[i]], ch0_present = ch0_present, min_pe = min_pe) for i in range(len(ch0))]
    inst_a6ch = np.array(inst_a6ch)
    if np.all(np.isnan(inst_a6ch)) : max_a6ch = np.nan
    else :
        max_a6ch = np.nanmax(inst_a6ch)
    return max_a6ch

def get_peaktime_a6ch(channels : list, ch0_present = False, ADC_MHz = 5000, min_pe = 2):
    t_samp = 1000/ADC_MHz
    ch0 = channels[0]
    chref = channels[1]
    ch60 = channels[2]
    ch120 = channels[3]
    ch180 = channels[4]
    ch240 = channels[5]
    ch300 = channels[6]
    inst_a6ch = [get_inst_Asimmetry_6ch(channels = [ch0[i], chref[i], ch60[i], ch120[i], ch180[i], ch240[i], ch300[i]], ch0_present = ch0_present, min_pe = min_pe) for i in range(len(ch0))]
    inst_a6ch = np.array(inst_a6ch)
    if np.all(np.isnan(inst_a6ch)) : peaktime_ns = np.nan
    else :
        max_indx = np.nanargmax(inst_a6ch)
        peaktime_ns = max_indx*t_samp
    return peaktime_ns

def get_r_charge_bar(channels : list, min_time = 0 , max_time = 100, ADC_MHz = 5000):

    if ADC_MHz is None:
    
        ch0 = sum(channels[0])
        chref = sum(channels[1])
        ch60 = sum(channels[2])
        ch120 = sum(channels[3])
        ch180 = sum(channels[4])
        ch240 = sum(channels[5])
        ch300 = sum(channels[6])

    else:
    
        t_samp = 1000/ADC_MHz
        max_indx = math.ceil(max_time/t_samp)
        min_indx = math.floor(min_time/t_samp)
        ch0 = sum(channels[0][min_indx:max_indx])
        chref = sum(channels[1][min_indx:max_indx])
        ch60 = sum(channels[2][min_indx:max_indx])
        ch120 = sum(channels[3][min_indx:max_indx])
        ch180 = sum(channels[4][min_indx:max_indx])
        ch240 = sum(channels[5][min_indx:max_indx])
        ch300 = sum(channels[6][min_indx:max_indx])
        
    xy_ch0_pos = np.array([0,0])
    xy_chref = np.array([1, 0])
    xy_ch60_pos = np.array([np.cos(60*np.pi/180),np.sin(60*np.pi/180)])
    xy_ch120_pos = np.array([np.cos(120*np.pi/180),np.sin(120*np.pi/180)])
    xy_ch180_pos = np.array([np.cos(180*np.pi/180),np.sin(180*np.pi/180)])
    xy_ch240_pos = np.array([np.cos(240*np.pi/180),np.sin(240*np.pi/180)])
    xy_ch300_pos = np.array([np.cos(300*np.pi/180),np.sin(300*np.pi/180)])
    xy_chs = [xy_ch0_pos,xy_chref,xy_ch60_pos,xy_ch120_pos,xy_ch180_pos,xy_ch240_pos,xy_ch300_pos]
    charges = np.array([ch0, chref, ch60, ch120, ch180, ch240, ch300])
    x_chs = [xy_chs[i][0] for i in range(len(xy_chs))]
    y_chs = [xy_chs[i][1] for i in range(len(xy_chs))]
    x_bar = (np.sum(x_chs*charges))/(np.sum(charges))
    y_bar = (np.sum(y_chs*charges))/(np.sum(charges))
    r = np.sqrt(x_bar**2 + y_bar**2)
    return r

def get_x_charge_bar(channels : list, min_time = 0 , max_time = 100, ADC_MHz = 1000):

    if ADC_MHz is None:

        ch0 = sum(channels[0])
        chref = sum(channels[1])
        ch60 = sum(channels[2])
        ch120 = sum(channels[3])
        ch180 = sum(channels[4])
        ch240 = sum(channels[5])
        ch300 = sum(channels[6])

    else:
        
        t_samp = 1000/ADC_MHz
        max_indx = math.ceil(max_time/t_samp)
        min_indx = math.floor(min_time/t_samp)
        ch0 = sum(channels[0][min_indx:max_indx])
        chref = sum(channels[1][min_indx:max_indx])
        ch60 = sum(channels[2][min_indx:max_indx])
        ch120 = sum(channels[3][min_indx:max_indx])
        ch180 = sum(channels[4][min_indx:max_indx])
        ch240 = sum(channels[5][min_indx:max_indx])
        ch300 = sum(channels[6][min_indx:max_indx])
        
    xy_ch0_pos = np.array([0,0])
    xy_chref = np.array([1, 0])
    xy_ch60_pos = np.array([np.cos(60*np.pi/180),np.sin(60*np.pi/180)])
    xy_ch120_pos = np.array([np.cos(120*np.pi/180),np.sin(120*np.pi/180)])
    xy_ch180_pos = np.array([np.cos(180*np.pi/180),np.sin(180*np.pi/180)])
    xy_ch240_pos = np.array([np.cos(240*np.pi/180),np.sin(240*np.pi/180)])
    xy_ch300_pos = np.array([np.cos(300*np.pi/180),np.sin(300*np.pi/180)])
    xy_chs = [xy_ch0_pos,xy_chref,xy_ch60_pos,xy_ch120_pos,xy_ch180_pos,xy_ch240_pos,xy_ch300_pos]
    charges = np.array([ch0, chref, ch60, ch120, ch180, ch240, ch300])
    x_chs = [xy_chs[i][0] for i in range(len(xy_chs))]
    if np.sum(charges) != 0 : x_bar = (np.sum(x_chs*charges))/(np.sum(charges))
    else : x_bar = np.nan
    return x_bar

def get_y_charge_bar(channels : list, min_time = 0 , max_time = 100, ADC_MHz = 1000):

    if ADC_MHz is None:

        ch0 = sum(channels[0])
        chref = sum(channels[1])
        ch60 = sum(channels[2])
        ch120 = sum(channels[3])
        ch180 = sum(channels[4])
        ch240 = sum(channels[5])
        ch300 = sum(channels[6])

    else: 
        t_samp = 1000/ADC_MHz
        max_indx = math.ceil(max_time/t_samp)
        min_indx = math.floor(min_time/t_samp)
        ch0 = sum(channels[0][min_indx:max_indx])
        chref = sum(channels[1][min_indx:max_indx])
        ch60 = sum(channels[2][min_indx:max_indx])
        ch120 = sum(channels[3][min_indx:max_indx])
        ch180 = sum(channels[4][min_indx:max_indx])
        ch240 = sum(channels[5][min_indx:max_indx])
        ch300 = sum(channels[6][min_indx:max_indx])
        
    xy_ch0_pos = np.array([0,0])
    xy_chref = np.array([1, 0])
    xy_ch60_pos = np.array([np.cos(60*np.pi/180),np.sin(60*np.pi/180)])
    xy_ch120_pos = np.array([np.cos(120*np.pi/180),np.sin(120*np.pi/180)])
    xy_ch180_pos = np.array([np.cos(180*np.pi/180),np.sin(180*np.pi/180)])
    xy_ch240_pos = np.array([np.cos(240*np.pi/180),np.sin(240*np.pi/180)])
    xy_ch300_pos = np.array([np.cos(300*np.pi/180),np.sin(300*np.pi/180)])
    xy_chs = [xy_ch0_pos,xy_chref,xy_ch60_pos,xy_ch120_pos,xy_ch180_pos,xy_ch240_pos,xy_ch300_pos]
    charges = np.array([ch0, chref, ch60, ch120, ch180, ch240, ch300])
    y_chs = [xy_chs[i][1] for i in range(len(xy_chs))]
    if np.sum(charges) != 0 : y_bar = (np.sum(y_chs*charges))/(np.sum(charges))
    else : y_bar = np.nan
    return y_bar

def get_tmax(channels : list, ADC_MHz = 5000):
    t_max = [np.nan for i in range(7)]
    t_samp = 1000/ADC_MHz
    ch_tr = [np.array(ch) for ch in channels]
    indx_max = [np.nanargmax(ch) for ch in ch_tr]
    for i in range(len(t_max)):
        if np.all(ch_tr[i] == 0):
            t_max[i] = np.nan
        else : t_max[i] = indx_max[i]*t_samp 
    return t_max

def get_tmean(channels : list, ADC_MHz = 5000):
    mean_times = [np.nan for i in range(7)]
    t_samp = 1000/ADC_MHz
    ch_tr = [np.array(ch) for ch in channels]
    for i in range(len(mean_times)):
        times = np.arange(len(ch_tr[i])) * t_samp
        if np.all(ch_tr[i] == 0):
            mean_times[i] = np.nan
        else : mean_times[i] = (np.sum(times*ch_tr[i]))/ (np.sum(ch_tr[i]))
    return mean_times

def get_tlast(channels : list, ADC_MHz = 5000):
    t_last = [np.nan for i in range(7)]
    t_samp = 1000/ADC_MHz
    ch_tr = [np.array(ch) for ch in channels]
    indx_last = [np.max(np.nonzero(ch)[0]) if np.any(ch != 0) else np.nan for ch in ch_tr]
    for i in range(len(t_last)):
        if np.all(ch_tr[i] == 0):
            t_last[i] = np.nan
        else : t_last[i] = indx_last[i]*t_samp 
    return t_last

def get_Asimmetry_hybmax_6ch(channels : list, min_time = 0 , max_time = 100, ch0_present = False, ADC_MHz = 1000):
    t_samp = 1000/ADC_MHz
    max_indx = math.ceil(max_time/t_samp)
    min_indx = math.floor(min_time/t_samp)
    
    ch0_tr = channels[0][min_indx:max_indx]
    chref_tr = channels[1][min_indx:max_indx]
    ch60_tr = channels[2][min_indx:max_indx]
    ch120_tr = channels[3][min_indx:max_indx]
    ch180_tr = channels[4][min_indx:max_indx]
    ch240_tr = channels[5][min_indx:max_indx]
    ch300_tr = channels[6][min_indx:max_indx]
    
    ch0 = sum(channels[0][min_indx:max_indx])
    chref = sum(channels[1][min_indx:max_indx])
    ch60 = sum(channels[2][min_indx:max_indx])
    ch120 = sum(channels[3][min_indx:max_indx])
    ch180 = sum(channels[4][min_indx:max_indx])
    ch240 = sum(channels[5][min_indx:max_indx])
    ch300 = sum(channels[6][min_indx:max_indx])
    
    ch_crop = [ch0_tr, chref_tr, ch60_tr, ch120_tr, ch180_tr, ch240_tr, ch300_tr]
    times_max = get_tmax(ch_crop, ADC_MHz = ADC_MHz)

    if not np.isnan(times_max[0]) : t0_max = max(times_max[0], 0.2)
    else : t0_max   = max_time
    if not np.isnan(times_max[1]) : tref_max = max(times_max[1], 0.2)
    else : tref_max = max_time
    if not np.isnan(times_max[2]) : t60_max  = max(times_max[2], 0.2)
    else : t60_max  = max_time
    if not np.isnan(times_max[3]) : t120_max = max(times_max[3], 0.2)
    else : t120_max = max_time
    if not np.isnan(times_max[4]) : t180_max = max(times_max[4], 0.2)
    else : t180_max = max_time
    if not np.isnan(times_max[5]) : t240_max = max(times_max[5], 0.2)
    else : t240_max = max_time
    if not np.isnan(times_max[6]) : t300_max = max(times_max[6], 0.2)
    else : t300_max = max_time
    
    front_pr = np.nan
    if ch0_present == False : front_pr = chref*1/tref_max + ch60*1/t60_max + ch300*1/t300_max
    else : front_pr = chref*1/tref_max + ch60*1/t60_max + ch300*1/t300_max + ch0*1/t0_max
    back_pr = ch120*1/t120_max + ch180*1/t180_max + ch240*1/t240_max
    if (front_pr + back_pr) != 0 : return (front_pr - back_pr)/(front_pr + back_pr)
    else : return np.nan

def get_Asimmetry_hyblast_6ch(channels : list, min_time = 0 , max_time = 100, ch0_present = False, ADC_MHz = 1000):
    t_samp = 1000/ADC_MHz
    max_indx = math.ceil(max_time/t_samp)
    min_indx = math.floor(min_time/t_samp)
    
    ch0_tr = channels[0][min_indx:max_indx]
    chref_tr = channels[1][min_indx:max_indx]
    ch60_tr = channels[2][min_indx:max_indx]
    ch120_tr = channels[3][min_indx:max_indx]
    ch180_tr = channels[4][min_indx:max_indx]
    ch240_tr = channels[5][min_indx:max_indx]
    ch300_tr = channels[6][min_indx:max_indx]
    
    ch0 = sum(channels[0][min_indx:max_indx])
    chref = sum(channels[1][min_indx:max_indx])
    ch60 = sum(channels[2][min_indx:max_indx])
    ch120 = sum(channels[3][min_indx:max_indx])
    ch180 = sum(channels[4][min_indx:max_indx])
    ch240 = sum(channels[5][min_indx:max_indx])
    ch300 = sum(channels[6][min_indx:max_indx])
    
    ch_crop = [ch0_tr, chref_tr, ch60_tr, ch120_tr, ch180_tr, ch240_tr, ch300_tr]
    times_last = get_tlast(ch_crop, ADC_MHz = ADC_MHz)

    if not np.isnan(times_last[0]): t0_last = max(times_last[0], 0.2)
    else: t0_last = max_time
    if not np.isnan(times_last[1]): tref_last = max(times_last[1], 0.2)
    else: tref_last = max_time
    if not np.isnan(times_last[2]): t60_last = max(times_last[2], 0.2) 
    else: t60_last = max_time
    if not np.isnan(times_last[3]): t120_last = max(times_last[3], 0.2)
    else: t120_last = max_time
    if not np.isnan(times_last[4]): t180_last = max(times_last[4], 0.2)
    else: t180_last = max_time
    if not np.isnan(times_last[5]): t240_last = max(times_last[5], 0.2)
    else: t240_last = max_time
    if not np.isnan(times_last[6]): t300_last = max(times_last[6], 0.2)
    else: t300_last = max_time

    
    front_pr = np.nan
    if ch0_present == False : front_pr = chref*1/tref_last + ch60*1/t60_last + ch300*1/t300_last
    else : front_pr = chref*1/tref_last + ch60*1/t60_last + ch300*1/t300_last + ch0*1/t0_last
    back_pr = ch120*1/t120_last + ch180*1/t180_last + ch240*1/t240_last
    if (front_pr + back_pr) != 0 : return (front_pr - back_pr)/(front_pr + back_pr)
    else : return np.nan

def get_Asimmetry_hybmax_2ch(channels : list, min_time = 0 , max_time = 100, ch0_present = False, ADC_MHz = 1000):
    t_samp = 1000/ADC_MHz
    max_indx = math.ceil(max_time/t_samp)
    min_indx = math.floor(min_time/t_samp)
    
    ch0_tr = channels[0][min_indx:max_indx]
    chref_tr = channels[1][min_indx:max_indx]
    ch60_tr = channels[2][min_indx:max_indx]
    ch120_tr = channels[3][min_indx:max_indx]
    ch180_tr = channels[4][min_indx:max_indx]
    ch240_tr = channels[5][min_indx:max_indx]
    ch300_tr = channels[6][min_indx:max_indx]
    
    ch0 = sum(channels[0][min_indx:max_indx])
    chref = sum(channels[1][min_indx:max_indx])
    ch60 = sum(channels[2][min_indx:max_indx])
    ch120 = sum(channels[3][min_indx:max_indx])
    ch180 = sum(channels[4][min_indx:max_indx])
    ch240 = sum(channels[5][min_indx:max_indx])
    ch300 = sum(channels[6][min_indx:max_indx])
    
    ch_crop = [ch0_tr, chref_tr, ch60_tr, ch120_tr, ch180_tr, ch240_tr, ch300_tr]
    times_max = get_tmax(ch_crop, ADC_MHz = ADC_MHz)

    if not np.isnan(times_max[0]) : t0_max = max(times_max[0], 0.2) 
    else : t0_max   = max_time
    if not np.isnan(times_max[1]) : tref_max = max(times_max[1], 0.2)
    else : tref_max = max_time
    if not np.isnan(times_max[2]) : t60_max  = max(times_max[2], 0.2) 
    else : t60_max  = max_time
    if not np.isnan(times_max[3]) : t120_max = max(times_max[3], 0.2) 
    else : t120_max = max_time
    if not np.isnan(times_max[4]) : t180_max = max(times_max[4], 0.2) 
    else : t180_max = max_time
    if not np.isnan(times_max[5]) : t240_max = max(times_max[5], 0.2) 
    else : t240_max = max_time
    if not np.isnan(times_max[6]) : t300_max = max(times_max[6], 0.2)
    else : t300_max = max_time
    
    front_pr = np.nan
    if ch0_present == False : front_pr = chref*1/tref_max 
    else : front_pr = chref*1/tref_max + ch0*1/t0_max
    back_pr = ch180*1/t180_max
    if (front_pr + back_pr) != 0 : return (front_pr - back_pr)/(front_pr + back_pr)
    else : return np.nan

def get_Asimmetry_hyblast_2ch(channels : list, min_time = 0 , max_time = 100, ch0_present = False, ADC_MHz = 1000):
    t_samp = 1000/ADC_MHz
    max_indx = math.ceil(max_time/t_samp)
    min_indx = math.floor(min_time/t_samp)
    
    ch0_tr = channels[0][min_indx:max_indx]
    chref_tr = channels[1][min_indx:max_indx]
    ch60_tr = channels[2][min_indx:max_indx]
    ch120_tr = channels[3][min_indx:max_indx]
    ch180_tr = channels[4][min_indx:max_indx]
    ch240_tr = channels[5][min_indx:max_indx]
    ch300_tr = channels[6][min_indx:max_indx]
    
    ch0 = sum(channels[0][min_indx:max_indx])
    chref = sum(channels[1][min_indx:max_indx])
    ch60 = sum(channels[2][min_indx:max_indx])
    ch120 = sum(channels[3][min_indx:max_indx])
    ch180 = sum(channels[4][min_indx:max_indx])
    ch240 = sum(channels[5][min_indx:max_indx])
    ch300 = sum(channels[6][min_indx:max_indx])
    
    ch_crop = [ch0_tr, chref_tr, ch60_tr, ch120_tr, ch180_tr, ch240_tr, ch300_tr]
    times_last = get_tlast(ch_crop, ADC_MHz = ADC_MHz)

    if not np.isnan(times_last[0]): t0_last = max(times_last[0], 0.2) 
    else: t0_last = max_time
    if not np.isnan(times_last[1]): tref_last = max(times_last[1], 0.2) 
    else: tref_last = max_time
    if not np.isnan(times_last[2]): t60_last = max(times_last[2], 0.2)
    else: t60_last = max_time
    if not np.isnan(times_last[3]): t120_last = max(times_last[3], 0.2) 
    else: t120_last = max_time
    if not np.isnan(times_last[4]): t180_last = max(times_last[4], 0.2)
    else: t180_last = max_time
    if not np.isnan(times_last[5]): t240_last = max(times_last[5], 0.2)
    else: t240_last = max_time
    if not np.isnan(times_last[6]): t300_last = max(times_last[6], 0.2) 
    else: t300_last = max_time
    
    front_pr = np.nan
    if ch0_present == False : front_pr = chref*1/tref_last
    else : front_pr = chref*1/tref_last + ch0*1/t0_last
    back_pr = ch180*1/t180_last
    if (front_pr + back_pr) != 0 : return (front_pr - back_pr)/(front_pr + back_pr)
    else : return np.nan

def get_Asimmetry_maxoplateau_6ch(channels : list, ch0_present = False, ADC_MHz = 1000, min_pe = 2):
    ch0 = channels[0]
    chref = channels[1]
    ch60 = channels[2]
    ch120 = channels[3]
    ch180 = channels[4]
    ch240 = channels[5]
    ch300 = channels[6]
    inst_a6ch = [get_inst_Asimmetry_6ch(channels = [ch0[i], chref[i], ch60[i], ch120[i], ch180[i], ch240[i], ch300[i]], ch0_present = ch0_present, min_pe = min_pe) for i in range(len(ch0))]
    inst_a6ch = np.nan_to_num(inst_a6ch, nan=0.0)
    a_max = get_peak_a6ch(channels, ch0_present = ch0_present, ADC_MHz = ADC_MHz, min_pe = 2)
    a_cum = np.cumsum(inst_a6ch) 
    a_pl = a_cum[-1]
    if a_pl != 0 : return a_max/a_pl
    else : return np.nan

def get_Asimmetry_maxoplateau_2ch(channels : list, ch0_present = False, ADC_MHz = 1000, min_pe = 2):
    ch0 = channels[0]
    chref = channels[1]
    ch60 = channels[2]
    ch120 = channels[3]
    ch180 = channels[4]
    ch240 = channels[5]
    ch300 = channels[6]
    inst_a2ch = [get_inst_Asimmetry_2ch(channels = [ch0[i], chref[i], ch60[i], ch120[i], ch180[i], ch240[i], ch300[i]], ch0_present = ch0_present, min_pe = min_pe) for i in range(len(ch0))]
    inst_a2ch = np.nan_to_num(inst_a2ch, nan=0.0)
    a_max = get_peak_a2ch(channels, ch0_present = ch0_present, ADC_MHz = ADC_MHz, min_pe = 2)
    a_cum = np.cumsum(inst_a2ch) 
    a_pl = a_cum[-1]
    if a_pl != 0 : return a_max/a_pl
    else : return np.nan

def get_Asimmetry_q50(channels:list,min_time=0,max_time=100,ch0_present=False,ADC_MHz=1000):
    t_samp=1000/ADC_MHz
    max_indx=math.ceil(max_time/t_samp)
    min_indx=math.floor(min_time/t_samp)
    t=np.arange(len(channels[0]))*t_samp
    t_crop=t[min_indx:max_indx]
    ch0_tr=channels[0][min_indx:max_indx]
    chref_tr=channels[1][min_indx:max_indx]
    ch60_tr=channels[2][min_indx:max_indx]
    ch120_tr=channels[3][min_indx:max_indx]
    ch180_tr=channels[4][min_indx:max_indx]
    ch240_tr=channels[5][min_indx:max_indx]
    ch300_tr=channels[6][min_indx:max_indx]
    if ch0_present==False: front_trace=chref_tr+ch60_tr+ch300_tr
    else: front_trace=ch0_tr+chref_tr+ch60_tr+ch300_tr
    back_trace=ch120_tr+ch240_tr+ch180_tr
    c_front=np.cumsum(front_trace)
    c_back=np.cumsum(back_trace)
    if c_front[-1]<=0 or c_back[-1]<=0: return np.nan
    try: t50_front=t_crop[np.searchsorted(c_front,0.5*c_front[-1])]
    except: t50_front=np.nan
    try: t50_back=t_crop[np.searchsorted(c_back,0.5*c_back[-1])]
    except: t50_back=np.nan
    if np.isnan(t50_front) or np.isnan(t50_back): return np.nan
    if (t50_front+t50_back)!=0: A_q50=(t50_front-t50_back)/(t50_front+t50_back)
    else: A_q50=np.nan
    return A_q50

def get_Asymmetry_rise(channels:list,min_time=0,max_time=100,ch0_present=False,ADC_MHz=1000,eps=1e-9):
    t_samp=1000/ADC_MHz
    max_indx=math.ceil(max_time/t_samp)
    min_indx=math.floor(min_time/t_samp)
    t=np.arange(len(channels[0]))*t_samp
    t_crop=t[min_indx:max_indx]
    ch0_tr   = np.array(channels[0][min_indx:max_indx])
    chref_tr = np.array(channels[1][min_indx:max_indx])
    ch60_tr  = np.array(channels[2][min_indx:max_indx])
    ch120_tr = np.array(channels[3][min_indx:max_indx])
    ch180_tr = np.array(channels[4][min_indx:max_indx])
    ch240_tr = np.array(channels[5][min_indx:max_indx])
    ch300_tr = np.array(channels[6][min_indx:max_indx])
    if ch0_present==False: front_trace=chref_tr+ch60_tr+ch300_tr
    else: front_trace=ch0_tr+chref_tr+ch60_tr+ch300_tr
    back_trace=ch120_tr+ch240_tr+ch180_tr
    if np.sum(front_trace)+np.sum(back_trace)==0: return np.nan
    num_inst=front_trace-back_trace
    den_inst=front_trace+back_trace+eps
    Acum=np.cumsum(num_inst)/np.cumsum(den_inst)
    if np.all(np.isnan(Acum)): return np.nan
    pmax=np.nanmax(Acum)
    i10=min(np.searchsorted(Acum,0.1*pmax), len(Acum)-1)
    i90=min(np.searchsorted(Acum,0.9*pmax), len(Acum)-1)
    A_rise=t_crop[i90]-t_crop[i10]
    return A_rise

def get_Asimmetry_60_240(channels : list, min_time = 0, max_time = 100, ch0_present=False, ADC_MHz=1000):
    t_samp = 1000/ADC_MHz
    max_indx = math.ceil(max_time/t_samp)
    min_indx = math.floor(min_time/t_samp)
    ch0 = sum(channels[0][min_indx:max_indx])
    chref = sum(channels[1][min_indx:max_indx])
    ch60 = sum(channels[2][min_indx:max_indx])
    ch120 = sum(channels[3][min_indx:max_indx])
    ch180 = sum(channels[4][min_indx:max_indx])
    ch240 = sum(channels[5][min_indx:max_indx])
    ch300 = sum(channels[6][min_indx:max_indx])
    front_ch = np.nan
    if ch0_present == False : front_ch = ch60 
    else : front_ch = ch60 + ch0
    back_ch = ch240
    if (front_ch + back_ch) != 0 : return (front_ch - back_ch)/(front_ch + back_ch)
    else : return np.nan    

def get_t_rise_10_50(channels : list, ADC_MHz = 1000):
    
    t_samp = 1000/ADC_MHz

    ch0 = channels[0]
    chref = channels[1]
    ch60 = channels[2]
    ch120 = channels[3]
    ch180 = channels[4]
    ch240 = channels[5]
    ch300 = channels[6]
    
    ch0_pe = sum(ch0)
    chref_pe = sum(chref)
    ch60_pe = sum(ch60)
    ch120_pe = sum(ch120)
    ch180_pe = sum(ch180)
    ch240_pe = sum(ch240)
    ch300_pe = sum(ch300)

    chs_pe = [ch0_pe, chref_pe, ch60_pe, ch120_pe, ch180_pe, ch240_pe, ch300_pe]

    def t_rise_diff(channel : list, channel_pe : int):
        
        ch_cum = np.cumsum(channel)
        i10 = np.searchsorted(ch_cum, 0.1*channel_pe)
        i50 = np.searchsorted(ch_cum, 0.5*channel_pe)
        i_diff = i50 - i10
        t_diff = t_samp * i_diff
        
        return t_diff

    t_rise_list = [t_rise_diff(channel, channel_pe) for channel, channel_pe in zip(channels, chs_pe)]
    
    return t_rise_list

def get_t_rise_50_90(channels : list, ADC_MHz = 1000):
    
    t_samp = 1000/ADC_MHz

    ch0 = channels[0]
    chref = channels[1]
    ch60 = channels[2]
    ch120 = channels[3]
    ch180 = channels[4]
    ch240 = channels[5]
    ch300 = channels[6]
    
    ch0_pe = sum(ch0)
    chref_pe = sum(chref)
    ch60_pe = sum(ch60)
    ch120_pe = sum(ch120)
    ch180_pe = sum(ch180)
    ch240_pe = sum(ch240)
    ch300_pe = sum(ch300)

    chs_pe = [ch0_pe, chref_pe, ch60_pe, ch120_pe, ch180_pe, ch240_pe, ch300_pe]

    def t_rise_diff(channel : list, channel_pe : int):
        
        ch_cum = np.cumsum(channel)
        i50 = np.searchsorted(ch_cum, 0.5*channel_pe)
        i90 = np.searchsorted(ch_cum, 0.9*channel_pe)
        i_diff = i90 - i50
        t_diff = t_samp * i_diff
        
        return t_diff

    t_rise_list = [t_rise_diff(channel, channel_pe) for channel, channel_pe in zip(channels, chs_pe)]
    
    return t_rise_list

def get_t_rise_10_90(channels : list, ADC_MHz = 1000):
    
    t_samp = 1000/ADC_MHz

    ch0 = channels[0]
    chref = channels[1]
    ch60 = channels[2]
    ch120 = channels[3]
    ch180 = channels[4]
    ch240 = channels[5]
    ch300 = channels[6]
    
    ch0_pe = sum(ch0)
    chref_pe = sum(chref)
    ch60_pe = sum(ch60)
    ch120_pe = sum(ch120)
    ch180_pe = sum(ch180)
    ch240_pe = sum(ch240)
    ch300_pe = sum(ch300)

    chs_pe = [ch0_pe, chref_pe, ch60_pe, ch120_pe, ch180_pe, ch240_pe, ch300_pe]

    def t_rise_diff(channel : list, channel_pe : int):
        
        ch_cum = np.cumsum(channel)
        i10 = np.searchsorted(ch_cum, 0.1*channel_pe)
        i90 = np.searchsorted(ch_cum, 0.9*channel_pe)
        i_diff = i90 - i10
        t_diff = t_samp * i_diff
        
        return t_diff

    t_rise_list = [t_rise_diff(channel, channel_pe) for channel, channel_pe in zip(channels, chs_pe)]
    
    return t_rise_list    

def get_Asimmetry_6ch_foractivechannels(channels : list, min_time = 0 , max_time = 100, ch0_present = False, ADC_MHz = 1000):
    active_chs = 0
    t_samp = 1000/ADC_MHz
    max_indx = math.ceil(max_time/t_samp)
    min_indx = math.floor(min_time/t_samp)
    ch0 = sum(channels[0][min_indx:max_indx])
    chref = sum(channels[1][min_indx:max_indx])
    ch60 = sum(channels[2][min_indx:max_indx])
    ch120 = sum(channels[3][min_indx:max_indx])
    ch180 = sum(channels[4][min_indx:max_indx])
    ch240 = sum(channels[5][min_indx:max_indx])
    ch300 = sum(channels[6][min_indx:max_indx])
    chs = [ch0, chref, ch60, ch120, ch180, ch240, ch300]
    pe = np.sum(chs)
    for i in chs:
        if i>=3:
            active_chs += 1
    front_ch = np.nan
    if ch0_present == False : front_ch = chref + ch60 + ch300
    else : front_ch = chref + ch60 + ch300 +ch0
    back_ch = ch120 + ch180 + ch240
    if (front_ch + back_ch) != 0 : return ((front_ch - back_ch - 1)/(front_ch + back_ch))*(active_chs-1)/6
    else : return np.nan