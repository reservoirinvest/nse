import click
from typing import Union
import pandas as pd
from ib_async import IB
from loguru import logger
from ibfuncs import get_open_orders, quick_pf
from nse import get_fnos, make_earliest_nse_nakeds
from utils import clean_symbols, pretty_print_df

def nse_nakeds(save: bool, fnos: Union[list, str, None]) -> pd.DataFrame:
    """Generates nakeds for NSE

    Args:
        save (bool): Pickles to `data/raw`
        fnos (Union[list, str, None]): If fno or a list doesn't save 

    Returns:
        pd.DataFrame: Naked options df
    """
    df = pd.DataFrame()  # Initialize df to avoid reference before assignment error
    
    if fnos: # prevents saving if fnos are given
        save = False
    
    fnos = get_fnos(fnos)

    # Make the nakeds
    try:
        df = make_earliest_nse_nakeds(fnos, save=save)
    except Exception as e:
        logger.error(f"Error in make_earliest_nakeds: {e}")
        df = pd.DataFrame([]) # empty df

    # Print a small sample upon success
    if not df.empty:
        df_print = df.drop(columns=['contract', 'expiry', 'instrument', 'ib_symbol'], errors='ignore')
        df_print = df_print.groupby('nse_symbol').head(2).iloc[:10]
        pretty_print_df(df_print)

    return df

def get_portfolio(port: int, clientId: int=10) -> pd.DataFrame:
    """Gets portfolio. Needs IB-TWS or IBG to be running.

    Args:
        port (int): `Live` port no
        clientId (int, optional): Client ID. Defaults to 10.

    Returns:
        pd.DataFrame: Portfolio df
    """
    with IB().connect(port=port, clientId=clientId) as ib:
        df = quick_pf(ib=ib)
        pretty_print_df(df)
    return df

def get_orders(symbols: Union[str, list, None], 
              active: bool, 
              port: int, 
              cid:int) -> pd.DataFrame:
    """Gets all open orders. Needs IB-TWS or IBG to be running.
    Args:
       active: if True shows only ACTIVE orders:   
       pending, pendingSubmit, presubmit and submitted
       port: Port of active IB client
       cid: Set as 10 for all API orders

    Returns:
       pd.DataFrame: Order df
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

@click.command()
@click.option('--function', type=click.Choice(['nse_nakeds', 'get_orders', 'get_portfolio'], case_sensitive=False), required=True, help='Function to execute.')
@click.option('--save', is_flag=True, default=True, help='Pickles to `data/raw` if set (only for nse_nakeds).')
@click.option('--fnos', type=str, multiple=True, help='FNOs as a list of strings or a single string. Use comma to separate multiple values (only for nse_nakeds).')
@click.option('--port', type=int, help='Port number for IB connection (required for get_orders and get_portfolio).')
@click.option('--clientid', type=int, default=10, help='Client ID for IB connection (default is 10).')
@click.option('--active', is_flag=True, default=False, help='If set, shows only ACTIVE orders (only for get_orders).')
@click.option('--symbols', type=str, multiple=True, help='Symbols to filter orders (only for get_orders).')
def cli(function, save, fnos, port, clientid, active, symbols):
    """Command line interface for IBKR option functions."""
    if function == 'nse_nakeds':
        # Convert fnos to a list if it is provided
        fnos_list = list(fnos) if fnos else None
        nse_nakeds(save, fnos_list)
    elif function == 'get_orders':
        if port is None:
            raise click.BadParameter('Port is required for get_orders.')
        symbols_list = list(symbols) if symbols else None
        get_orders(symbols_list, active, port, clientid)
    elif function == 'get_portfolio':
        if port is None:
            raise click.BadParameter('Port is required for get_portfolio.')
        get_portfolio(port, clientid)
    
    # Print or process the result as needed
    # print(result)

if __name__ == '__main__':
    cli()