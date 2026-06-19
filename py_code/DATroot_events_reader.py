import uproot
import pandas as pd
import numpy as np
from py_code import DATroot_primary_reader as dat
import os
import pyarrow as pa, pyarrow.parquet as pq, gc
from py_code import reco_regression

def Read_Roots_For_Training(DATname : str, path_to_DAT_root : str, fit_type : str, output_dir_smu : str, output_dir_nmu : str, max_time : float, max_energy : float, max_theta : float, min_energy : float = 100*(10**3), min_theta : float = 0, verbose : bool = False):

    if not os.path.exists(path_to_DAT_root):
        return pd.DataFrame(), pd.DataFrame()
    else:
        if verbose :
            print("Working on ", path_to_DAT_root)
        
    

    with uproot.open(path_to_DAT_root) as DAT:
        
        if "XCDF" in DAT:
            tree = DAT["XCDF"]
        else:
            if verbose : 
                print(f"Skipping {path_to_DAT_root}: 'XCDF' not found")
            return pd.DataFrame()
                
    #DAT = uproot.open(path_to_DAT_root)
    #tree = DAT["XCDF"]
    df = tree.arrays(library="pd")

    evts_number = len(df["HAWCSim.Evt.Num"].unique())
    selected_prims = 0

    for i in df["HAWCSim.Evt.Num"].unique():
        
        p = df[df["HAWCSim.Evt.Num"] == i] 
        p_theta = np.asarray(p['HAWCSim.Evt.Theta'].iloc[0])
        p_energy = np.asarray(p['HAWCSim.Evt.Energy'].iloc[0])
        p_X = np.asarray(p['HAWCSim.Evt.X'].iloc[0])
        p_Y = np.asarray(p['HAWCSim.Evt.Y'].iloc[0])
        p_R = np.sqrt(p_X**2 + p_Y**2)

        ## grieco_reco ##
        PE_True = np.asarray(p["HAWCSim.Evt.nPE"].iloc[0])
        E_reco = reco_regression.E_reco(PE_True, fit_type)*1000 #in GeV
        ####
        
        #if (p_theta > max_theta) or (p_energy > max_energy) or (p_theta < min_theta) or (p_energy < min_energy):
            #continue
        
        if (p_theta > max_theta) or (E_reco > max_energy) or (p_theta < min_theta) or (E_reco < min_energy):
            continue
            
        else:

            #print("E_reco [TeV] = ", E_reco/1000)

            p = dat.get_stations_info(p, "../survey_and_array_txt_repo/tank_pos_H_4FF.txt")
            if p is None:
                continue

            p = dat.get_shower_time(p) 
            check_trigger_time = p['Trigger_Time'].iloc[0]
            if np.isnan(check_trigger_time.any()):
                continue

            selected_prims += 1
                
            p = dat.get_dt_correction(p)
            p = dat.appy_time_correction(p)
        
            p_sm = dat.get_st_lvl_train_sm_df(p, max_time = max_time)
            p_nmu = dat.get_st_lvl_train_nmu_df(p, max_time = max_time)

            output_dir_smu  = output_dir_smu 
            output_dir_nmu = output_dir_nmu
            
            out_path_smu = f"{output_dir_smu}{DATname}_{i}.parquet"
            table_smu = pa.Table.from_pandas(p_sm)
            pq.write_table(table_smu, out_path_smu, compression="zstd")

            out_path_nmu = f"{output_dir_nmu}{DATname}_{i}.parquet"
            table_nmu = pa.Table.from_pandas(p_nmu)
            pq.write_table(table_nmu, out_path_nmu, compression="zstd")

    print(DATname + ", Number of primaries in this DAT ", evts_number, "-> Primaries after the cut = ", selected_prims)
    if selected_prims != 0:
        return p_sm, p_nmu
    else:
        empty_df = pd.DataFrame()
        return empty_df, empty_df
                
def Read_Roots_For_Testing(output_dir : str, DATname : str, path_to_DAT_root: str, fit_type : str, max_time : float, max_energy : float, max_theta : float, min_energy : float = 100*(10**3), min_theta : float = 0, verbose : bool = False):

    if not os.path.exists(path_to_DAT_root):
        return pd.DataFrame(), pd.DataFrame()
    else:
        if verbose:
            print("Working on ", path_to_DAT_root)

    try:
        with uproot.open(path_to_DAT_root) as DAT:
            
            if "XCDF" not in DAT:
                print(f"Skipping {path_to_DAT_root}: no XCDF")
                return pd.DataFrame()
    
            tree = DAT["XCDF"]
    
    except Exception as e:
        print(f"Skipping {path_to_DAT_root}: {e}")
        return pd.DataFrame()
        
    #DAT = uproot.open(path_to_DAT_root)
    #tree = DAT["XCDF"]
    df = tree.arrays(library="pd")

    evts_number = len(df["HAWCSim.Evt.Num"].unique())
    selected_prims = 0

    for i in df["HAWCSim.Evt.Num"].unique():
        
        p = df[df["HAWCSim.Evt.Num"] == i] 
        p_theta = np.asarray(p['HAWCSim.Evt.Theta'].iloc[0])
        p_energy = np.asarray(p['HAWCSim.Evt.Energy'].iloc[0])
        p_X = np.asarray(p['HAWCSim.Evt.X'].iloc[0])
        p_Y = np.asarray(p['HAWCSim.Evt.Y'].iloc[0])
        p_R = np.sqrt(p_X**2 + p_Y**2)

        ## grieco_reco ##
        PE_True = np.asarray(p["HAWCSim.Evt.nPE"].iloc[0])
        E_reco = reco_regression.E_reco(PE_True, fit_type)*1000 #in GeV
        ####
        
        if (p_theta > max_theta) or (E_reco > max_energy) or (p_theta < min_theta) or (E_reco < min_energy):
            continue
            
        else:

            #print("E_reco [TeV] = ", E_reco/1000)
            
            p = dat.get_stations_info(p, "../survey_and_array_txt_repo/tank_pos_H_4FF.txt")
            if p is None:
                continue

            p = dat.get_shower_time(p)
            check_trigger_time = p['Trigger_Time'].iloc[0]
            if np.isnan(check_trigger_time.any()):
                continue

            p = dat.get_dt_correction(p)
            p = dat.appy_time_correction(p)

            selected_prims += 1
        
            p_test = dat.get_st_lvl_test_df(p, max_time = max_time)
            
            out_path = f"{output_dir}{DATname}_{i}.parquet"
            table = pa.Table.from_pandas(p_test)
            pq.write_table(table, out_path, compression="zstd")

    if selected_prims == 0:

        print("Sorry, no primaries after the cut :(")
        return pd.DataFrame()

    else :
        
        print(DATname + ", Number of primaries in this DAT ", evts_number, "-> Primaries after the cut = ", selected_prims)
    
        return p_test
