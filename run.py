import click
from from_root import from_root
from ib_async import IB
from loguru import logger

from ibfuncs import get_open_orders, quick_pf
from nse import NSEfnos
from utils import get_files_from_patterns, load_config

config = load_config()

ROOT = from_root()
PORT = port = config.get("PORT")


# Set the loguru logger
logger.add(sink=ROOT / "log" / "run.log", mode="w", level=config.get('LOGLEVEL'))


# --- FUNCTIONS AND CHOICES

@click.group()
def cli():
    """NSE command line interface"""

# --- for open orders ---
@cli.command(name='get-open-orders')
@click.option('--active', default=False, help='Shows only orders with ACTIVE status: pending, pendingSubmit, presubmit and submitted')
def open_ords(active):
    with IB().connect(port=port, clientId=10) as ib:
        df = get_open_orders(ib=ib, is_active=active)
        print(df)

# --- for portfolio ---
@cli.command(name='get-portfolio', help='Gets current portfolio from ib')
def get_portfolio():
    with IB().connect(port=port, clientId=10) as ib:
        print(quick_pf(ib=ib))


if __name__ == "__main__":
    cli()