import pandas as pd
import numpy as np


def get_stations_info(primary: pd.DataFrame, tank_pos_file_path : str, number_of_channels: int = 31381, channels_per_stations: int = 7):

    if len(primary) != 1:
        return None
    else : 
        primary = primary.copy()
        
    PEs_ID = np.asarray(primary["HAWCSim.PE.PMTID"].iloc[0])
    Station = (PEs_ID // channels_per_stations).astype(int)

    max_station = number_of_channels // channels_per_stations
    Station = np.clip(Station, 0, max_station - 1)

    primary["PE.Station_ID"] = pd.Series([Station], index=primary.index, dtype=object)

    data_channels = np.loadtxt(fname = tank_pos_file_path)
    chID, xT, yT, rT = data_channels[:, 0], data_channels[:, 1], data_channels[:, 2], data_channels[:, 3]

    ch_IDposX_map = dict(zip(chID, xT))
    ch_IDposY_map = dict(zip(chID, yT))
    ch_IDposR_map = dict(zip(chID, rT))

    PE_ID_posX = np.array([ch_IDposX_map[pe] for pe in PEs_ID])
    PE_ID_posY = np.array([ch_IDposY_map[pe] for pe in PEs_ID])
    PE_ID_posR = np.array([ch_IDposR_map[pe] for pe in PEs_ID])

    primary["PE.Station_X"] = pd.Series([PE_ID_posX], index=primary.index, dtype=object)
    primary["PE.Station_Y"] = pd.Series([PE_ID_posY], index=primary.index, dtype=object)
    primary["PE.Station_R"] = pd.Series([PE_ID_posR], index=primary.index, dtype=object)      

    return primary

def get_shower_time(primary : pd.DataFrame):
    
    dummy = primary.copy()
    PE_xT = np.asarray(dummy["PE.Station_X"].iloc[0])
    PE_yT = np.asarray(dummy["PE.Station_Y"].iloc[0])
    pe_times  = np.asarray(dummy["HAWCSim.PE.Time"].iloc[0])
    p_X = np.asarray(dummy['HAWCSim.Evt.X'].iloc[0])
    p_Y = np.asarray(dummy['HAWCSim.Evt.Y'].iloc[0])
    p_R = np.sqrt(p_X**2 + p_Y**2)
    mask = np.sqrt((p_X - PE_xT)**2 + (p_Y - PE_yT)**2) <= 30*100 
    filtered_times = pe_times[mask]
    if len(filtered_times) != 0:
        min_time = np.min(filtered_times)
    else :
        min_time = np.nan        
    trigger_times = np.full(len(pe_times), min_time)
    primary["Trigger_Time"] = pd.Series([trigger_times], index = primary.index, dtype = object)

    return primary

def get_dt_correction(primary : pd.DataFrame):

    dummy = primary.copy()
    station_r = np.asarray(dummy["PE.Station_R"].iloc[0])
    PE_xT = np.asarray(dummy["PE.Station_X"].iloc[0])
    PE_yT = np.asarray(dummy["PE.Station_Y"].iloc[0])
    p_X = np.asarray(dummy['HAWCSim.Evt.X'].iloc[0])
    p_Y = np.asarray(dummy['HAWCSim.Evt.Y'].iloc[0])
    p_theta = np.asarray(dummy['HAWCSim.Evt.Theta'].iloc[0])
    p_phi = np.asarray(dummy['HAWCSim.Evt.Phi'].iloc[0])
    station_phi = np.atan2(PE_yT - p_Y, PE_xT - p_X)
    p_R = np.sqrt(p_X**2 + p_Y**2)
    tank_to_core = np.sqrt((p_X - PE_xT)**2 + (p_Y - PE_yT)**2)
    c = 29.9792 #cm/ns
    dt = (tank_to_core/c)*np.sin(p_theta*np.pi/180)*np.cos(p_phi*np.pi/180 - station_phi)
    primary["Time_Correction"] = pd.Series([dt], index = primary.index, dtype = object)

    return primary

def appy_time_correction(primary : pd.DataFrame):

    dummy = primary.copy()
    pe_times  = np.asarray(dummy["HAWCSim.PE.Time"].iloc[0])
    trigger_time = np.asarray(dummy["Trigger_Time"].iloc[0])
    dt = np.asarray(dummy["Time_Correction"].iloc[0])
    pe_times_corrected = pe_times - trigger_time + dt
    primary["PE.Corrected_Times"] = pd.Series([pe_times_corrected], index = dummy.index, dtype = object)

    return primary  

def make_trace(times, max_time):
    
    arr = np.asarray(times, dtype=int)
    arr = arr[(arr >= 0) & (arr < max_time)]

    return np.bincount(arr, minlength=max_time)[:max_time].astype(int).tolist()

def get_st_lvl_test_df(primary : pd.DataFrame, number_of_channels: int = 31381, channels_per_stations: int = 7, max_time:float = 100):

    dummy = primary.copy()

    energies = []
    thetas = []
    XCore = []
    YCore = []
    ch0 = []
    ch1 = []
    ch2 = []
    ch3 = []
    ch4 = []
    ch5 = []
    ch6 = []
    x_pe = []
    y_pe = []
    r_pe = []
    t_from_c = []
    orig_pe = []
    pe_frac_mu = []
    aretheremuon = []

    #nominal info on the primary
    p_Energy = np.asarray(dummy['HAWCSim.Evt.Energy'].iloc[0])
    p_theta = np.asarray(dummy['HAWCSim.Evt.Theta'].iloc[0])
    p_X = np.asarray(dummy['HAWCSim.Evt.X'].iloc[0]) 
    p_Y = np.asarray(dummy['HAWCSim.Evt.Y'].iloc[0])

    for i in range(0, number_of_channels, channels_per_stations):
        
        times_pe_ch0 = []
        times_pe_ch1 = []
        times_pe_ch2 = []
        times_pe_ch3 = []
        times_pe_ch4 = []
        times_pe_ch5 = []
        times_pe_ch6 = []
        
        ch0_trace = []
        ch1_trace = []
        ch2_trace = []
        ch3_trace = []
        ch4_trace = []
        ch5_trace = []
        ch6_trace = []

        istheremuon = 0
        pe_by_muon = 0
        pe_not_by_muon = 0
        
        PEs_ID = np.asarray(dummy["HAWCSim.PE.PMTID"].iloc[0])
        sh_plane_times = np.asarray(dummy["PE.Corrected_Times"].iloc[0])
        PE_origin = np.asarray(dummy["HAWCSim.PE.origPType"].iloc[0])

        mask = (PEs_ID < i + channels_per_stations) & (PEs_ID >= i) & (sh_plane_times <= max_time)
        
        #pes for that PMTs in that station and times within 100ns
        pe_times_ = np.asarray(dummy["PE.Corrected_Times"].iloc[0])[mask]
        PEs_ID_ = np.asarray(dummy["HAWCSim.PE.PMTID"].iloc[0])[mask]
        PE_origin_ = np.asarray(dummy["HAWCSim.PE.origPType"].iloc[0])[mask]

        for pe_time, pe_id, pe_nature in zip(pe_times_, PEs_ID_, PE_origin_):
            
            if pe_id == i:
                times_pe_ch0.append(pe_time)
            elif pe_id == i +1:
                times_pe_ch1.append(pe_time)
            elif pe_id == i + 2:
                times_pe_ch2.append(pe_time)
            elif pe_id == i + 3:
                times_pe_ch3.append(pe_time)
            elif pe_id == i + 4:
                times_pe_ch4.append(pe_time)
            elif pe_id == i + 5:
                times_pe_ch5.append(pe_time)
            elif pe_id == i + 6:
                times_pe_ch6.append(pe_time)
                
            if pe_nature == 5 or pe_nature == 6 :
                istheremuon = 1
                pe_by_muon += 1
            else :
                pe_not_by_muon +=1
                
        ch0_trace = make_trace(times_pe_ch0, max_time)
        ch1_trace = make_trace(times_pe_ch1, max_time)
        ch2_trace = make_trace(times_pe_ch2, max_time)
        ch3_trace = make_trace(times_pe_ch3, max_time)
        ch4_trace = make_trace(times_pe_ch4, max_time)
        ch5_trace = make_trace(times_pe_ch5, max_time)
        ch6_trace = make_trace(times_pe_ch6, max_time)
        
        #tank position for that station
        PE_xT = np.asarray(dummy["PE.Station_X"].iloc[0])[mask]
        PE_yT = np.asarray(dummy["PE.Station_Y"].iloc[0])[mask]
        PE_rT = np.asarray(dummy["PE.Station_R"].iloc[0])[mask]
        T_from_C = np.sqrt((PE_xT - p_X)**2 + (PE_yT - p_Y)**2)

        #percentage of pe from muons
        if (pe_by_muon + pe_not_by_muon) > 0:
            frac_mu = pe_by_muon/(pe_by_muon + pe_not_by_muon)
        else :
            frac_mu = 0
               
        #particle origin of each PE

        energies.append(np.round(p_Energy, 1))  
        thetas.append(np.round(p_theta, 1))     
        XCore.append(np.round(p_X, 1))          
        YCore.append(np.round(p_Y, 1))          
        x_pe.append(float(np.round(PE_xT[0]/100, 1)) if len(PE_xT) > 0 else np.nan)
        y_pe.append(float(np.round(PE_yT[0]/100, 1)) if len(PE_yT) > 0 else np.nan)
        r_pe.append(float(np.round(PE_rT[0]/100, 1)) if len(PE_rT) > 0 else np.nan)
        t_from_c.append(float(np.round(T_from_C[0]/100, 1)) if len(T_from_C) > 0 else np.nan)
        orig_pe.append(PE_origin)
        ch0.append(ch0_trace)
        ch1.append(ch1_trace)
        ch2.append(ch2_trace)
        ch3.append(ch3_trace)
        ch4.append(ch4_trace)
        ch5.append(ch5_trace)
        ch6.append(ch6_trace)
        pe_frac_mu.append(frac_mu)
        aretheremuon.append(istheremuon)

    df_event_level = pd.DataFrame({

        "ch0": ch0,
        "ch1": ch1,
        "ch2": ch2,
        "ch3": ch3,
        "ch4": ch4,
        "ch5": ch5,
        "ch6": ch6,
        "PE_XTank": x_pe,
        "PE_YTank": y_pe, 
        "PE_RTank": r_pe,
        "Nominal_Energy": energies,
        "Nominal_Theta": thetas,
        "Nominal_XCore": XCore,
        "Nominal_YCore": YCore,
        "T_C" : t_from_c,
        "frac_mu" : pe_frac_mu,
        "IsThereMuon" : aretheremuon
    })

    return df_event_level

def get_st_lvl_train_sm_df(primary : pd.DataFrame, number_of_channels: int = 31381, channels_per_stations: int = 7, max_time:float = 100):

    dummy = primary.copy()

    energies = []
    thetas = []
    XCore = []
    YCore = []
    ch0 = []
    ch1 = []
    ch2 = []
    ch3 = []
    ch4 = []
    ch5 = []
    ch6 = []
    x_pe = []
    y_pe = []
    r_pe = []
    t_from_c = []
    orig_pe = []
    pe_frac_mu = []
    aretheremuon = []

    #nominal info on the primary
    p_Energy = np.asarray(dummy['HAWCSim.Evt.Energy'].iloc[0])
    p_theta = np.asarray(dummy['HAWCSim.Evt.Theta'].iloc[0])
    p_X = np.asarray(dummy['HAWCSim.Evt.X'].iloc[0])
    p_Y = np.asarray(dummy['HAWCSim.Evt.Y'].iloc[0])

    for i in range(0, number_of_channels, channels_per_stations):
        
        times_pe_ch0 = []
        times_pe_ch1 = []
        times_pe_ch2 = []
        times_pe_ch3 = []
        times_pe_ch4 = []
        times_pe_ch5 = []
        times_pe_ch6 = []
        
        ch0_trace = []
        ch1_trace = []
        ch2_trace = []
        ch3_trace = []
        ch4_trace = []
        ch5_trace = []
        ch6_trace = []

        istheremuon = 0
        pe_by_muon = 0
        pe_not_by_muon = 0
        
        PEs_ID = np.asarray(dummy["HAWCSim.PE.PMTID"].iloc[0])
        sh_plane_times = np.asarray(dummy["PE.Corrected_Times"].iloc[0])
        PE_origin = np.asarray(dummy["HAWCSim.PE.origPType"].iloc[0])
        trackID  = np.asarray(dummy["HAWCSim.PE.origTrackID"].iloc[0])


        #In case of cut on tank distances add the following lines
        #PE_xT_all = np.asarray(dummy["PE.Station_X"].iloc[0])
        #PE_yT_all = np.asarray(dummy["PE.Station_Y"].iloc[0])
        #distance2 = (PE_xT_all - p_X)**2 + (PE_yT_all - p_Y)**2 #as definition
        #(distance2 <= (300*100)**2) #in the mask

        mask = (PEs_ID < i + channels_per_stations) & (PEs_ID >= i) & (sh_plane_times <= max_time) & ((PE_origin == 5) | (PE_origin == 6)) 

        one_muon_condition = np.unique(trackID[mask]) #ensuring there is not more than one muon

        if len(one_muon_condition) != 1:
            continue
            
        #pes for that PMTs in that station and times within 100ns
        pe_times_ = np.asarray(dummy["PE.Corrected_Times"].iloc[0])[mask]
        PEs_ID_ = np.asarray(dummy["HAWCSim.PE.PMTID"].iloc[0])[mask]
        PE_origin_ = np.asarray(dummy["HAWCSim.PE.origPType"].iloc[0])[mask]

        for pe_time, pe_id, pe_nature in zip(pe_times_, PEs_ID_, PE_origin_):
            
            if pe_id == i:
                times_pe_ch0.append(pe_time)
            elif pe_id == i +1:
                times_pe_ch1.append(pe_time)
            elif pe_id == i + 2:
                times_pe_ch2.append(pe_time)
            elif pe_id == i + 3:
                times_pe_ch3.append(pe_time)
            elif pe_id == i + 4:
                times_pe_ch4.append(pe_time)
            elif pe_id == i + 5:
                times_pe_ch5.append(pe_time)
            elif pe_id == i + 6:
                times_pe_ch6.append(pe_time)
                
            if pe_nature == 5 or pe_nature == 6 :
                istheremuon = 1
                pe_by_muon += 1
            else :
                pe_not_by_muon +=1
                
        ch0_trace = make_trace(times_pe_ch0, max_time)
        ch1_trace = make_trace(times_pe_ch1, max_time)
        ch2_trace = make_trace(times_pe_ch2, max_time)
        ch3_trace = make_trace(times_pe_ch3, max_time)
        ch4_trace = make_trace(times_pe_ch4, max_time)
        ch5_trace = make_trace(times_pe_ch5, max_time)
        ch6_trace = make_trace(times_pe_ch6, max_time)
        
        #tank position for that station
        PE_xT = np.asarray(dummy["PE.Station_X"].iloc[0])[mask]
        PE_yT = np.asarray(dummy["PE.Station_Y"].iloc[0])[mask]
        PE_rT = np.asarray(dummy["PE.Station_R"].iloc[0])[mask]
        T_from_C = np.sqrt((PE_xT - p_X)**2 + (PE_yT - p_Y)**2)

        #percentage of pe from muons
        if (pe_by_muon + pe_not_by_muon) > 0:
            frac_mu = pe_by_muon/(pe_by_muon + pe_not_by_muon)
        else :
            frac_mu = 0
        
        #particle origin of each PE

        energies.append(np.round(p_Energy, 1))  
        thetas.append(np.round(p_theta, 1))     
        XCore.append(np.round(p_X, 1))          
        YCore.append(np.round(p_Y, 1))          
        x_pe.append(float(np.round(PE_xT[0]/100, 1)) if len(PE_xT) > 0 else np.nan)
        y_pe.append(float(np.round(PE_yT[0]/100, 1)) if len(PE_yT) > 0 else np.nan)
        r_pe.append(float(np.round(PE_rT[0]/100, 1)) if len(PE_rT) > 0 else np.nan)
        t_from_c.append(float(np.round(T_from_C[0]/100, 1)) if len(T_from_C) > 0 else np.nan)
        ch0.append(ch0_trace)
        ch1.append(ch1_trace)
        ch2.append(ch2_trace)
        ch3.append(ch3_trace)
        ch4.append(ch4_trace)
        ch5.append(ch5_trace)
        ch6.append(ch6_trace)
        pe_frac_mu.append(frac_mu)
        aretheremuon.append(istheremuon)

    df_event_level = pd.DataFrame({

        "ch0": ch0,
        "ch1": ch1,
        "ch2": ch2,
        "ch3": ch3,
        "ch4": ch4,
        "ch5": ch5,
        "ch6": ch6,
        "PE_XTank": x_pe,
        "PE_YTank": y_pe, 
        "PE_RTank": r_pe,
        "Nominal_Energy": energies,
        "Nominal_Theta": thetas,
        "Nominal_XCore": XCore,
        "Nominal_YCore": YCore,
        "T_C" : t_from_c,
        "frac_mu" : pe_frac_mu,
        "IsThereMuon" : aretheremuon
    })

    return df_event_level

def get_st_lvl_train_nmu_df(primary : pd.DataFrame, number_of_channels: int = 31381, channels_per_stations: int = 7, max_time:float = 100):

    dummy = primary.copy()

    energies = []
    thetas = []
    XCore = []
    YCore = []
    ch0 = []
    ch1 = []
    ch2 = []
    ch3 = []
    ch4 = []
    ch5 = []
    ch6 = []
    x_pe = []
    y_pe = []
    r_pe = []
    orig_pe = []
    t_from_c = []
    pe_frac_mu = []
    aretheremuon = []

    #nominal info on the primary
    p_Energy = np.asarray(dummy['HAWCSim.Evt.Energy'].iloc[0])
    p_theta = np.asarray(dummy['HAWCSim.Evt.Theta'].iloc[0])
    p_X = np.asarray(dummy['HAWCSim.Evt.X'].iloc[0])
    p_Y = np.asarray(dummy['HAWCSim.Evt.Y'].iloc[0])

    for i in range(0, number_of_channels, channels_per_stations):
        
        times_pe_ch0 = []
        times_pe_ch1 = []
        times_pe_ch2 = []
        times_pe_ch3 = []
        times_pe_ch4 = []
        times_pe_ch5 = []
        times_pe_ch6 = []
        
        ch0_trace = []
        ch1_trace = []
        ch2_trace = []
        ch3_trace = []
        ch4_trace = []
        ch5_trace = []
        ch6_trace = []

        istheremuon = 0
        pe_by_muon = 0
        pe_not_by_muon = 0
        
        PEs_ID = np.asarray(dummy["HAWCSim.PE.PMTID"].iloc[0])
        sh_plane_times = np.asarray(dummy["PE.Corrected_Times"].iloc[0])
        PE_origin = np.asarray(dummy["HAWCSim.PE.origPType"].iloc[0])

        #In case of cut on tank distances add the following lines
        #PE_xT_all = np.asarray(dummy["PE.Station_X"].iloc[0])
        #PE_yT_all = np.asarray(dummy["PE.Station_Y"].iloc[0])
        #distance2 = (PE_xT_all - p_X)**2 + (PE_yT_all - p_Y)**2  #as definition
        #(distance2 <= (300*100)**2) #in the mask
  
        mask_check = (PEs_ID < i + channels_per_stations) & (PEs_ID >= i) & (sh_plane_times <= max_time)
        PE_origin_check = PE_origin[mask_check]
        
        if np.any((PE_origin_check == 5) | (PE_origin_check == 6)):
            continue
        
        mask = (PEs_ID < i + channels_per_stations) & (PEs_ID >= i) & (sh_plane_times <= max_time) & ((PE_origin != 5) & (PE_origin != 6)) 

        #pes for that PMTs in that station and times within 100ns
        pe_times_ = np.asarray(dummy["PE.Corrected_Times"].iloc[0])[mask]
        PEs_ID_ = np.asarray(dummy["HAWCSim.PE.PMTID"].iloc[0])[mask]
        PE_origin_ = np.asarray(dummy["HAWCSim.PE.origPType"].iloc[0])[mask]

        for pe_time, pe_id, pe_nature in zip(pe_times_, PEs_ID_, PE_origin_):
            
            if pe_id == i:
                times_pe_ch0.append(pe_time)
            elif pe_id == i +1:
                times_pe_ch1.append(pe_time)
            elif pe_id == i + 2:
                times_pe_ch2.append(pe_time)
            elif pe_id == i + 3:
                times_pe_ch3.append(pe_time)
            elif pe_id == i + 4:
                times_pe_ch4.append(pe_time)
            elif pe_id == i + 5:
                times_pe_ch5.append(pe_time)
            elif pe_id == i + 6:
                times_pe_ch6.append(pe_time)

            if pe_nature == 5 or pe_nature == 6 :
                istheremuon = 1
                pe_by_muon += 1
            else :
                pe_not_by_muon +=1
                
        ch0_trace = make_trace(times_pe_ch0, max_time)
        ch1_trace = make_trace(times_pe_ch1, max_time)
        ch2_trace = make_trace(times_pe_ch2, max_time)
        ch3_trace = make_trace(times_pe_ch3, max_time)
        ch4_trace = make_trace(times_pe_ch4, max_time)
        ch5_trace = make_trace(times_pe_ch5, max_time)
        ch6_trace = make_trace(times_pe_ch6, max_time)
        
        #tank position for that station
        PE_xT = np.asarray(dummy["PE.Station_X"].iloc[0])[mask]
        PE_yT = np.asarray(dummy["PE.Station_Y"].iloc[0])[mask]
        PE_rT = np.asarray(dummy["PE.Station_R"].iloc[0])[mask]
        T_from_C = np.sqrt((PE_xT - p_X)**2 + (PE_yT - p_Y)**2)

        #percentage of pe from muons
        if (pe_by_muon + pe_not_by_muon) > 0:
            frac_mu = pe_by_muon/(pe_by_muon + pe_not_by_muon)
        else :
            frac_mu = 0
        
        #particle origin of each PE

        energies.append(np.round(p_Energy, 1))  
        thetas.append(np.round(p_theta, 1))     
        XCore.append(np.round(p_X, 1))          
        YCore.append(np.round(p_Y, 1))          
        x_pe.append(float(np.round(PE_xT[0]/100, 1)) if len(PE_xT) > 0 else np.nan)
        y_pe.append(float(np.round(PE_yT[0]/100, 1)) if len(PE_yT) > 0 else np.nan)
        r_pe.append(float(np.round(PE_rT[0]/100, 1)) if len(PE_rT) > 0 else np.nan)
        t_from_c.append(float(np.round(T_from_C[0]/100, 1)) if len(T_from_C) > 0 else np.nan)
        ch0.append(ch0_trace)
        ch1.append(ch1_trace)
        ch2.append(ch2_trace)
        ch3.append(ch3_trace)
        ch4.append(ch4_trace)
        ch5.append(ch5_trace)
        ch6.append(ch6_trace)
        pe_frac_mu.append(frac_mu)
        aretheremuon.append(istheremuon)

    df_event_level = pd.DataFrame({

        "ch0": ch0,
        "ch1": ch1,
        "ch2": ch2,
        "ch3": ch3,
        "ch4": ch4,
        "ch5": ch5,
        "ch6": ch6,
        "PE_XTank": x_pe,
        "PE_YTank": y_pe, 
        "PE_RTank": r_pe,
        "Nominal_Energy": energies,
        "Nominal_Theta": thetas,
        "Nominal_XCore": XCore,
        "Nominal_YCore": YCore,
        "T_C" : t_from_c,
        "frac_mu" : pe_frac_mu,
        "IsThereMuon" : aretheremuon
    })

    return df_event_level