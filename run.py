# --- CLI RUN

import re
import sys
from pathlib import Path

import click
import pandas as pd
from from_root import from_root
from ib_async import IB
from loguru import logger

from nse import NSEfnos, make_earliest_nakeds, nse_ban_list, get_all_fno_names
from utils import (delete_files, get_files_from_patterns, pretty_print_df,
                   split_and_uppercase, yes_or_no, get_pickle)

# Set the root
ROOT = from_root()

# Set path for imported modules
def set_module_path(ROOT: Path):
     
    if str(ROOT) not in sys.path:
                sys.path.insert(1, str(ROOT))

    # Add `src` and ROOT to _src.pth in .venv to allow imports in VS Code
    from pathlib import Path
    from sysconfig import get_path

    if "src" not in Path.cwd().parts:
        src_path = str(Path(get_path("purelib")) / "_src.pth")
        with open(src_path, "w") as f:
            f.write(str(ROOT / "src\n"))
            f.write(str(ROOT))

set_module_path(ROOT=ROOT)

# ---- IMPORT MY MODULES -----
# ----------------------------

from ibfuncs import get_open_orders, quick_pf
from nse import NSEfnos
from utils import clean_symbols, get_files_from_patterns, load_config

# load configs and set the logger
config = load_config()
logger.add(sink=ROOT / "log" / "run.log", mode="w", level=config.get('LOGLEVEL'))

# --- CONSTANTS ---
# -----------------

PORT = port = config.get("PORT")
CID = config.get("CLIENTID")
NSE2IB = config.get('NSE2IB')


@click.group()
def cli():
    """NSE command line interface"""
    pass

# --- CLICK CHOICES ---
# ----------------------

# *--- to make earliest nakeds ---

@cli.command(name='ib-early-nakeds', help='Makes earliest nakeds')
@click.option('--save', default=False, is_flag=True)
# @click.option('--fnos', default=[], multiple=True, help='List of FNOS (comma-separated).')
@click.argument('fnos', type=str, nargs=-1, required=False)
def make_nakeds(save, fnos):
    """Makes and shows naked options for earliest dte
    Args:
       save: True pickles in data/raw folder"""

    if not fnos:
        files = get_files_from_patterns(ROOT/'data'/ 'raw')
        if files:
            ans = yes_or_no("Delete remenants of earliest?")
            fnos = get_all_fno_names()

            if ans: # Delete the files!
                delete_files(files)
            else:
                p = [get_pickle(f) for f in files]

                # remove the remnant symbols from fnos
                remove = set(pd.concat(p, axis=0, ignore_index=True).nse_symbol.unique())
                fnos = fnos - remove

        # nse = NSEfnos()
        # fnos = list(nse.equities())

    else:
        fnos = split_and_uppercase(fnos)
        
    if not isinstance(fnos, list):
        fnos = list(fnos)

    fnos.sort # sort the list

    df = make_earliest_nakeds(fnos, save=save)

    # print a small sample
    df_print = df.drop(columns=['contract', 'expiry', 
                                'instrument', 'ib_symbol'],
                        errors='ignore')
    
    if not df_print.empty:
        df_print = df_print.groupby('nse_symbol').head(2).head(10)

    pretty_print_df(df_print)

    return df


# *--- for open orders ---

@cli.command(name='ib-open-orders', help='Shows open orders from IB.')
@click.argument('symbols', type=str, nargs=-1, required=False)
@click.option('--active', default=False, help='Needs an active TWS-IB/IBG connection')
@click.option('--port', default=3000, help='Active IB port')
@click.option('--cid', default=10, help='Active IB Client ID')

def open_ords(symbols, active, port, cid) -> pd.DataFrame:
    """ NOTE: Needs IB-TWS or IBG to be running.
    Args:
       active: if True shows only ACTIVE orders:   
       pending, pendingSubmit, presubmit and submitted
       port: Port of active IB client
       clientId: Set as 10 for all API orders

    Returns:
       df
    """
    with IB().connect(port=port, clientId=cid) as ib:
        df = get_open_orders(ib=ib, is_active=active)
        df = df.drop(columns=['contract', 'order'], errors='ignore')

        # clean up the symbols if provided
        if symbols:
            symbols = clean_symbols(symbols)
            df = df[df.symbol.isin(symbols)]

        pretty_print_df(df)

    return df


# *--- for portfolio ---

@cli.command(name='ib-portfolio', help='Shows current portfolio from IB.')
@click.option('--port', default=3000, help='Active IB port')
@click.option('--clientId', default=10, help='Active IB Client ID')
def get_portfolio(port, clientId):
    """ NOTE: Needs IB-TWS or IBG to be running.
    Args:
       port: Port of active IB client
       clientId: Set as 10 for all API orders

    Returns:
       df
    """
    with IB().connect(port=port, clientId=clientId) as ib:
        df = print(quick_pf(ib=ib))

        pretty_print_df(df)

    return df

if __name__ == "__main__":
    cli()