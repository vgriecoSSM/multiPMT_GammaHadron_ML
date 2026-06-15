ENERGY_RECO_PARAMS = {
    
    "H3FF": {"m": 429.74, "q": -6.507e3},
    "K3FF": {"m": 310.8,  "q": -5730.2},
    "H4FF": {"m": 556.74, "q": -9.578e3}
    
                     }

def E_reco(PE, fit_type):
    
    try:
        
        m = ENERGY_RECO_PARAMS[fit_type]["m"]
        q = ENERGY_RECO_PARAMS[fit_type]["q"]
        
    except KeyError:
        raise ValueError(f"Invalid fit_type '{fit_type}'. Expected one of {list(ENERGY_RECO_PARAMS)}")
    
    return (PE - q) / m
        

