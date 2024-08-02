# ---- UNNECESSARY CODE NOT CONTRIBUTING ANYTHING----


import pandas as pd
from loguru import logger
from tqdm import tqdm

from nse import NSEfnos, equity_iv_df


def make_raw_fno_df(fnos) -> pd.DataFrame:
    """Makes all the raw fnos"""

    n = NSEfnos()

    dfs = []

    with tqdm(total=len(fnos), desc='Generating raw fnos', unit='symbol') as pbar:
        
        for s in fnos:
            try:
                df = equity_iv_df(n.stock_quote_fno(s))
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error for {s} - error: {e}")
                pass

            pbar.update(1)
            
    df = pd.concat(dfs, ignore_index=True)

    return df
