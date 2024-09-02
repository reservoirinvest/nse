# --- SNP SPECIFIC FUNCTIONS ---
# ===============================

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from from_root import from_root
from ib_async import Index, Stock
from ibfuncs import get_ib, get_mkt_prices, qualify_me
from utils import clean_ib_util_df, get_pickle, load_config, pickle_me
import pandas_datareader.data as web

MARKET = 'SNP'
ROOT = from_root()

config = load_config(MARKET=MARKET)

# ---- SETTING CONSTANTS ----
PUTSTDMULT = config.get("PUTSTDMULT")
CALLSTDMULT = config.get("CALLSTDMULT")

# PORT = port = config.get("PORT")

indexes_path = ROOT / "data" / "templates" / "snp_indexes.yml"

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
        FRESH (bool, optional): Regenerates underlyings. Defaults to False.

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

    r = us_repo_rate()
    print(f"repo_rate is {r} that is of type{type(r)}")

