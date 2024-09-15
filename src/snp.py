# --- SNP SPECIFIC FUNCTIONS ---
# ===============================

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yaml
from from_root import from_root
from ib_async import IB, Contract, Index, Stock, util
from loguru import logger

from ibfuncs import (get_a_price_iv, get_ib, get_mkt_prices, get_open_orders,
                     make_chains, make_ib_orders, place_orders,
                     process_in_chunks, qualify_me, quick_pf)
from utils import (Timer, append_black_scholes, append_safe_strikes,
                   append_xPrice, arrange_orders, clean_ib_util_df,
                   get_closest_strike, get_pickle, how_many_days_old,
                   load_config, pickle_me, strip_split, yes_or_no)

MARKET = 'SNP'
ROOT = from_root()

config = load_config(MARKET=MARKET)

# ---- SETTING CONSTANTS ----
PUTSTDMULT = config.get("PUTSTDMULT")
CALLSTDMULT = config.get("CALLSTDMULT")
MAXDTE = config.get("MAXDTE")
MINEXPROM = config.get("MINEXPROM")


# PORT = port = config.get("PORT")

indexes_path = ROOT / "data" / "templates" / "snp_indexes.yml"


# *--- CORE SNP FUNCTIONS ---
def make_snp_naked_puts(save: bool=False):
    timer = Timer("Making snp nakeds")
    timer.start()

    # Assemble snp unds and save
    df_unds = assemble_snp_underlyings(FRESH=True)
    unds_path = ROOT / 'data' / 'snp_unds.pkl'
    pickle_me(df_unds, unds_path)

    # Make chains
    df_ch = make_chains(df_unds, save=True)

    # Make put targets from chains closest to strike and < MINDTE
    dfp = df_ch[df_ch.right == "P"]
    dfp = dfp[dfp.dte <= MAXDTE]

    dfe = dfp[dfp.groupby(['ib_symbol', 'dte']).dte.transform('min').astype('int') == dfp.dte.astype('int')]
    dfef = dfe[~dfe.undPrice.isnull()]

    dft = dfef.groupby(['ib_symbol', 'dte']) \
            .apply(lambda x: get_closest_strike(x), include_groups=False) \
            .reset_index().set_index('level_2') \
            .rename_axis('')

    dft = dft.sort_values(['ib_symbol', 'dte'])

    # Compute the IV to be average of historical and implied, if implied is less than historical
    min_series = pd.Series(np.minimum(dft.und_iv, dft.und_hv))
    weighted_avg_series = (dft.und_iv + dft.und_hv) / 2 * 0.75
    iv = pd.Series(np.where(dft.und_iv < dft.und_hv, min_series + weighted_avg_series, dft.und_iv), index=dft.index)
    dft = dft.assign(iv=iv)

    # ...remove null ivs
    dft = dft.loc[~dft.iv.isnull()]

    # Get the safe strikes
    dft = append_safe_strikes(dft, PUTSTDMULT, CALLSTDMULT)

    # Get black scholes price
    risk_free_rate = us_repo_rate() / 100
    dft = append_black_scholes(dft, risk_free_rate)


    # Get the market prices
    # ...build contracts

    contracts = [Contract('OPT', symbol=s, lastTradeDateOrContractMonth=util.formatIBDatetime(e)[:8], strike=k, right=r,
                        exchange='SMART', currency='USD')
                        for s, e, k, r
                        in zip(dft.ib_symbol, dft.expiry, dft.strike, dft.right)]


    # ...qualify contracts
    with get_ib(MARKET) as ib:
        cts = ib.run(process_in_chunks(ib, contracts, func=qualify_me, func_args={'desc': 'qualified'}, chunk_size=200, chunk_desc="Qualifying..."))


    # ...get market price for contracts
    with get_ib(MARKET) as ib:
        res = ib.run(process_in_chunks(ib, cts, func=get_a_price_iv, func_args={'sleep': 15, 'gentick':''}, chunk_desc='Pricing'))

    df_price = pd.concat(res, ignore_index=True).drop(['secType', 'iv', 'hv'], axis=1)
    dfn = df_price.merge(dft, on=['ib_symbol', 'expiry', 'strike', 'right'])
    df = snp_marcom(dfn)
    df = df[df.price>0] # remove zero price

    df_snp = append_xPrice(df.assign(lot=100), MINEXPROM)
    df_snp = df_snp.reset_index(drop=True)
    df_snp = df_snp.assign(lot=1) # reset lots for snp orders

    if save:
        pickle_me(df_snp, ROOT/'data'/'snp_nakeds.pkl')

    timer.stop()
    return df_snp

def place_snp_orders():
    MARKET = "SNP"

    pd.set_option('display.precision', 2)

    # Load configuration
    config = load_config(MARKET)
    port = config.get('PORT')
    MARGINPERORDER = config.get('MARGINPERORDER')

    # Check age of pickles and load data
    nakeds_path = ROOT / 'data' / 'snp_nakeds.pkl'
    txt = f'snp_nakeds.pkl is {how_many_days_old(nakeds_path): 0.2f} days old. Want to load??'
    if not yes_or_no(txt):
        print('Aborting order process.')
        return None

    df_opts = get_pickle(nakeds_path)
    cols = strip_split('ib_symbol,undPrice,strike,safe_strike,right,dte,bsPrice,price,xPrice,margin,rom')

    # Check open orders and positions
    with IB().connect(port=port, clientId=10) as ib:
        dfo = get_open_orders(ib)
        dfp = quick_pf(ib)

    remove_opens = set(dfo.symbol.to_list()) if not dfo.empty else set()
    remove_positions = set(dfp.symbol.to_list()) if not dfp.empty else set()
    remove_ib_syms = remove_opens | remove_positions

    # Get target options to place
    dft = df_opts[~df_opts.ib_symbol.isin(remove_ib_syms)].reset_index(drop=True)
    print(f'\n{len(dft)} options available for order placement\n')
    print(dft[cols].head())

    # Arrange and make orders
    df_nakeds = arrange_orders(dft, maxmargin=MARGINPERORDER)
    cos = make_ib_orders(df_nakeds)

    # Confirm order placement
    if not yes_or_no(f"Do you want to place {len(cos)} orders?"):
        print("Order placement aborted.")
        return df_nakeds

    # Place the orders
    with IB().connect(port=port, clientId=10) as ib:
        ordered = place_orders(ib=ib, cos=cos)

    # Save the ordered data
    filename = f"{datetime.now().strftime('%Y%m%d_%I_%M_%p')}_snp_naked_orders.pkl"
    pickle_me(ordered, ROOT / "data" / "xn_history" / filename)

    logger.info(f"{len(ordered)} Orders placed and saved successfully.")
    print(util.df(ordered).head())

# * --- ASSEMBLE ---------

def read_weeklys() -> pd.DataFrame:
    """gets weekly cboe symbols"""

    dls = "http://www.cboe.com/products/weeklys-options/available-weeklys"
    df = pd.read_html(dls)[0]

    return df


def rename_weekly_columns(df: pd.DataFrame) -> pd.DataFrame:
    """standardizes column names of cboe"""

    df.columns = ["desc", "symbol"]

    return df


def remove_non_char_symbols(df: pd.DataFrame) -> pd.DataFrame:
    """removes symbols with non-chars - like dots (BRK.B)"""

    df = df[df.symbol.str.extract("([^a-zA-Z])").isna()[0]]

    return df


def make_weekly_cboes() -> pd.DataFrame:
    """
    Generates a weekly cboe symbols dataframe
    """

    df = read_weeklys().pipe(rename_weekly_columns).pipe(remove_non_char_symbols)

    # add exchange
    df = df.assign(exchange="SMART")

    return df


def get_snps() -> pd.Series:
    """
    gets snp symbols from wikipedia
    """
    snp_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    snps = pd.read_html(snp_url)[0]["Symbol"]
    return snps


def add_snp_indexes(df: pd.DataFrame, path_to_yaml_file: str) -> pd.DataFrame:
    """
    add indexes from config/snp_indexes.yaml
    """
    with open(path_to_yaml_file, "r") as f:
        kv_pairs = yaml.load(f, Loader=yaml.FullLoader)

    dfs = []
    for k in kv_pairs.keys():
        dfs.append(
            pd.DataFrame(list(kv_pairs[k].items()), columns=["symbol", "desc"]).assign(
                exchange=k
            )
        )

    more_df = pd.concat(dfs, ignore_index=True)

    df_all = pd.concat([df, more_df], ignore_index=True)

    return df_all


def split_snp_stocks_and_index(df: pd.DataFrame) -> pd.DataFrame:
    """differentiates stocks and index"""

    df = df.assign(secType=np.where(df.desc.str.contains("Index"), "IND", "STK"))

    return df


def make_snp_weeklies(indexes_path: Path):
    """Makes snp weeklies with indexes"""

    # get snp stock weeklies
    df_weekly_cboes = make_weekly_cboes()
    snps = get_snps()

    # filter weekly snps
    df_weekly_snps = df_weekly_cboes[df_weekly_cboes.symbol.isin(snps)].reset_index(
        drop=True
    )

    # add index weeklies
    df_weeklies = add_snp_indexes(df_weekly_snps, indexes_path).pipe(
        split_snp_stocks_and_index
    )

    return df_weeklies


def make_unqualified_snp_underlyings(df: pd.DataFrame) -> pd.DataFrame:
    """Build underlying contracts"""

    contracts = [
        Stock(symbol=symbol, exchange=exchange, currency="USD")
        if secType == "STK"
        else Index(symbol=symbol, exchange=exchange, currency="USD")
        for symbol, secType, exchange in zip(df.symbol, df.secType, df.exchange)
    ]

    df = df.assign(contract=contracts)

    return df


def assemble_snp_underlyings(LIVE: bool=True,
                                FRESH: bool=True) -> pd.DataFrame:
    """Assembles a df of SNP underlying contracts

    Args:
        LIVE (bool, optional): True=LIVE | False=PAPER. Defaults to True.
        FRESH (bool, optional): Regenerates underlyings. Defaults to True.

    Returns:
        pd.DataFrame: _description_
    """

    undpath = ROOT/'data'/'snp_unds.pkl'
    df = get_pickle(undpath)

    if df is None or df.empty or FRESH:
        df = make_snp_weeklies(indexes_path).pipe(make_unqualified_snp_underlyings)

        contracts = df.contract.to_list()

        with get_ib(MARKET='snp', LIVE=LIVE) as ib:
            qualified_contracts = ib.run(qualify_me(ib,
                                                contracts,
                                                desc='Qualifying SNP Unds'))

            dfc = clean_ib_util_df(qualified_contracts)
            df = ib.run(get_mkt_prices(ib,
                                       dfc.contract,
                                       sleep=15,
                                       chunk_size=39))
            df.rename(columns={'conId': 'undId', 'iv': 'und_iv',
                               'hv': 'und_hv', 'price': 'undPrice'}, inplace=True)

        pickle_me(df, undpath)

    return df


def us_repo_rate():
    """Risk free US interest rate

    Returns:
        _type_: float (5.51)

    """
    tbill_yield = web.DataReader('DGS1MO', 'fred', start=datetime.now() -
                                 timedelta(days=365), end=datetime.now())['DGS1MO'].iloc[-1]
    return tbill_yield


def snp_marcom(df:pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the maintenance margin and commissions for IBKR options using a DataFrame.

    Parameters:
    - df: A pandas DataFrame with columns 'strike', 'undPrice', 'price', and 'right'.

    Returns:
    - df: The original DataFrame with additional columns for maintenance margin and commission.
    """

    # Calculate the in-the-money amount
    df['itm'] = df.apply(lambda row:
                                         max(row['strike'] - row['undPrice'], 0) if row['right'] == 'P'
                                         else max(row['undPrice'] - row['strike'], 0), axis=1)

    # Calculate maintenance margin based on option type
    df['margin'] = df.apply(lambda row:
                                        row['price'] + max(0.20 * (2 * row['undPrice']) - row['itm'], 0.10 * row['strike'])
                                        if row['right'] == 'P'
                                        else row['price'] + max(0.15 * (3 * row['undPrice']) - row['itm'], 0.10 * row['strike']), axis=1)

    # Adjust for lot size
    df['margin'] *= 100

    # Calculate commissions (fixed rate per contract of 100 lots)
    commission_per_contract = 0.65
    df['comm'] = commission_per_contract  # Fixed commission for each contract

    df.drop(columns='itm', inplace=True)

    # Return the updated DataFrame
    return df


if __name__ == "__main__":

    df = make_snp_naked_puts(save=True)
    print(f'{len(df)}contracts found!!!')
    print(df.head())

