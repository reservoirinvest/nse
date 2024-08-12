# --- PLACES NSE NAKED ORDERS ---- 

import pandas as pd
from utils import load_config



def place_nakeds(MARKET:str) -> pd.DataFrame:

    # --- Set constants ---
    config = load_config(MARKET)
    pass