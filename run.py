import os
from typing import List, Optional, Union

import click
import pandas as pd
from ib_async import IB
from loguru import logger

from ibfuncs import account_values, get_open_orders, quick_pf
from nse import get_fnos, make_earliest_nse_nakeds, order_nse_nakeds
from snp import make_snp_naked_puts, place_snp_orders
from utils import choose_market, clean_symbols, get_port, pretty_print_df

# Ensure the log directory exists
log_dir = './log'
os.makedirs(log_dir, exist_ok=True)

# Configure logger to log to a file named after the script
logger.add(f"{log_dir}/run.log", rotation="1 MB")

def nse_nakeds(save: bool, fnos: Optional[Union[list, str]] = None) -> pd.DataFrame:
    """Generates nakeds for NSE"""
    if fnos:
        save = False

    fnos = get_fnos(fnos)

    try:
        df = make_earliest_nse_nakeds(fnos, save=save)
        if not df.empty:
            df_print = df.drop(columns=['contract', 'expiry', 'instrument', 'ib_symbol'], errors='ignore')
            pretty_print_df(df_print.groupby('nse_symbol').head(2).iloc[:10])
    except Exception as e:
        logger.error(f"Error in make_earliest_nakeds: {e}")
        df = pd.DataFrame()

    return df

def snp_nakeds(save: bool = True) -> pd.DataFrame:
    """Generates naked puts for SNP"""
    try:
        df = make_snp_naked_puts(save=save)
        if not df.empty:
            df_print = df.drop(columns=['contract', 'expiry', 'instrument', 'ib_symbol'], errors='ignore')
            pretty_print_df(df_print.head(10))
    except Exception as e:
        logger.error(f"Error in make_snp_naked_puts: {e}")
        df = pd.DataFrame()

    return df

def order_snp_nakeds():
    """Prepares and places SNP naked put orders"""

    place_snp_orders()

    # try:
    #     df_nakeds, cos = place_snp_orders()
    #     if df_nakeds is not None and cos is not None:
    #         place_snp_orders(df_nakeds, cos)
    #     else:
    #         print("No orders to place.")
    # except Exception as e:
    #     logger.error(f"Error in order_snp_nakeds: {e}")

def get_portfolio(port: int, clientId: int = 10) -> pd.DataFrame:
    """Gets portfolio. Needs IB-TWS or IBG to be running."""
    with IB().connect(port=port, clientId=clientId) as ib:
        df = quick_pf(ib=ib).drop(columns='contract')

    return df

def get_nlv(port: int, clientId: int = 10) -> dict:
    """Gets NLV, cushion and margins"""
    with IB().connect(port=port, clientId=clientId) as ib:
        nlv = ib.run(account_values(ib))
    return nlv

def get_orders(symbols: Optional[Union[str, list]] = None,
               active: bool = False,
               port: int = None,
               cid: int = 10) -> pd.DataFrame:
    """Gets all open orders. Needs IB-TWS or IBG to be running."""
    if not port:
        port = get_port(choose_market().upper())
    with IB().connect(port=port, clientId=cid) as ib:
        df = get_open_orders(ib=ib, is_active=active).drop(columns=['contract', 'order'], errors='ignore')

        if symbols:
            df = df[df.symbol.isin(clean_symbols(symbols))]

        # Limit to first 5 rows
        df_display = df.head(5)
        pretty_print_df(df_display)
    return df

def market_selection():
    """Prompt the user to select a market.

    Returns:
        str: Selected market
    """
    click.echo("Select the market:")
    click.echo("1. NSE")
    click.echo("2. SNP")
    choice = click.prompt("Enter your choice (1 or 2)", type=int)

    if choice == 1:
        return 'nse'
    elif choice == 2:
        return 'snp'
    else:
        click.echo("Invalid choice. Please select 1 or 2.")
        return market_selection()

@click.group()
def cli():
    """Command line interface for IBKR option functions."""
    pass

@cli.command()
@click.option('--save', is_flag=True, default=True, help='Pickle results to `data/raw`.')
@click.option('--fnos', type=str, multiple=True, help='FNOs as a list of strings or a single string. Use comma to separate multiple values.')
def nse_naked_options(save, fnos):
    """Generate nakeds for NSE."""
    nse_nakeds(save, list(fnos) if fnos else None)

@cli.command()
@click.option('--save', is_flag=True, default=True, help='Pickle results to `data/raw`.')
def snp_naked_puts(save):
    """Generate naked puts for SNP."""
    snp_nakeds(save)

@cli.command()
def nse_order_place():
    """Place NSE naked orders."""
    order_nse_nakeds()

@cli.command()
def snp_order_place():
    """Place SNP naked put orders."""
    place_snp_orders()

@cli.command()
@click.option('--market', type=click.Choice(['snp', 'nse'], case_sensitive=False), help='Choose market for port')
@click.option('--clientid', type=int, default=10, help='Client ID for IB connection (default is 10).')
@click.option('--active', is_flag=True, default=False, help='If set, shows only ACTIVE orders.')
@click.option('--symbols', type=str, multiple=True, help='Symbols to filter orders.')
def openorders(market, clientid, active, symbols):
    """Get open orders."""
    if not market:
        market = choose_market()
    port = get_port(market.upper())
    get_orders(list(symbols) if symbols else None, active, port, clientid)

@cli.command()
@click.option('--market', type=click.Choice(['snp', 'nse'], case_sensitive=False), help='Choose market for port')
@click.option('--clientid', type=int, default=10, help='Client ID for IB connection (default is 10).')
def portfolio(market, clientid):
    """Get portfolio."""
    if not market:
        market = choose_market()
    port = get_port(market.upper())
    result = get_portfolio(port, clientid)
    if isinstance(result, pd.DataFrame) and not result.empty:
        pretty_print_df(result)

@cli.command()
@click.option('--market', type=click.Choice(['snp', 'nse'], case_sensitive=False), help='Choose market for port')
@click.option('--clientid', type=int, default=10, help='Client ID for IB connection (default is 10).')
def nlv(market, clientid):
    """Get NLV, cushion and margins."""
    if not market:
        market = choose_market()
    port = get_port(market.upper())
    result = get_nlv(port, clientid)
    print(result)

if __name__ == '__main__':
    cli()
