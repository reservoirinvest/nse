
# Produce nakeds

# --- IMPORTS ---
# ---------------

import asyncio
import glob
import io
import json
import math
import pickle
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from itertools import chain
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import pytz
import requests
from bs4 import BeautifulSoup
from from_root import from_root
from ib_async import IB, MarketOrder, Option, Order, util
from loguru import logger
from pandas import json_normalize
from scipy.integrate import quad
from scipy.stats import norm
from tqdm import tqdm

from nse import (NSEfnos, Timer, all_early_fnos, get_files_from_patterns,
                 get_open_orders, get_pickle, nse_ban_list)


# --- CONSTANTS ---
# -----------------

live_port = 3000  # nse
paper_port = 3001  # nse paper trade

port = PORT = live_port

PUTSTDMULT = 1.8
CALLSTDMULT = 2.2


# Symbol Maps
IDX_SYM_HIST_MAP = {"BANKNIFTY": "Nifty Bank", "NIFTY": "Nifty 50"}
MARKET = "NSE"
ROOT = from_root()



timer = Timer("Earliest nakeds")
timer.start()

# Initialize FnO class
nse = NSEfnos()

# get executed symbols from pickles
files = get_files_from_patterns("/*nakeds*")
symbols_list = [get_pickle(ROOT / "data" / file).nse_symbol.to_list() for file in files]
executed = set(chain.from_iterable(symbols_list))

# get open order symbols
with IB().connect(port=port, clientId=10) as ib:

    trades = ib.reqAllOpenOrders()
    df_openords = get_open_orders(ib)
    open_orders = set(df_openords.symbol.to_list())

# get the fnos from pickles - if available

try:
    df = pd.concat(
        [get_pickle(f) for f in files],
        ignore_index=True,
    )

    # Sort by safe_strike: strike ratio

    # ... group calls
    gc = (
        df[df.right == "C"]
        .assign(ratio=df.safe_strike / df.strike)
        .sort_values("ratio")
        .groupby("ib_symbol")
    )
    df_calls = gc.head(2).sort_values(["ib_symbol", "strike"], ascending=[True, False])
    dfc = df_calls[df_calls.margin < 300000].reset_index(drop=True)
    dfc = dfc.assign(ratio=dfc.safe_strike / dfc.strike).sort_values("ratio")

    # ... group puts
    gc = (
        df[df.right == "P"]
        .assign(ratio=df.strike / df.safe_strike)
        .sort_values("ratio")
        .groupby("ib_symbol")
    )
    df_puts = gc.head(2).sort_values(["ib_symbol", "strike"], ascending=[True, False])
    dfp = df_puts[df_puts.margin < 300000].reset_index(drop=True)
    dfp = dfp.assign(ratio=dfp.strike / dfp.safe_strike).sort_values("ratio")

    # ... prepare the nakeds to order
    df_nakeds = pd.concat([dfc, dfp], axis=0, ignore_index=True)

except ValueError:
    pass

# build the fnos list
fnos = ((set(["BANKNIFTY", "NIFTY"]) | nse.equities()) - executed) - open_orders
fnos = fnos - set(nse_ban_list())    

# Build the earliest naked options

if fnos:
    df_nakeds = all_early_fnos(fnos, save=True)

print(df_nakeds.head())

timer.stop()