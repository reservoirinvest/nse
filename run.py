# --- CLI RUN

import sys
from pathlib import Path

import click
import pandas as pd
from from_root import from_root
from ib_async import IB
from loguru import logger
from tabulate import tabulate

# Set pandas options
pd.options.display.max_columns = None
# pd.options.display.max_rows = None
pd.set_option('display.precision', 2)

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

from ibfuncs import get_open_orders, quick_pf
from nse import NSEfnos
from utils import clean_symbols, get_files_from_patterns, load_config

# load configs and set the logger
config = load_config()
logger.add(sink=ROOT / "log" / "run.log", mode="w", level=config.get('LOGLEVEL'))

# --- CONSTANTS ---

PORT = port = config.get("PORT")
CID = config.get("CLIENTID")
NSE2IB = config.get('NSE2IB')

# --- CLICK FUNCTIONS ---

def pretty_print_df(df):
  """Pretty prints a pandas DataFrame to the console."""
  print(tabulate(df, headers='keys', tablefmt='pretty'))

# --- CLICK CHOICES ---

@click.group()
def cli():
    """NSE command line interface"""
    pass

# --- for open orders ---
@cli.command(name='ib-get-open-orders', help='Shows open orders from IB.')
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

# --- for portfolio ---
@cli.command(name='ib-get-portfolio', help='Shows current portfolio from IB.')
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
        print(quick_pf(ib=ib))


if __name__ == "__main__":
    cli()