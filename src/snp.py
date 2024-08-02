# --- SNP SPECIFIC FUNCTIONS ---
# ===============================

import asyncio
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from from_root import from_root
from ib_async import IB, Index, Stock
from tqdm.asyncio import tqdm_asyncio

from ibfuncs import qualify_me as ib_qualify_me
from utils import load_config as utils_load_config, to_list as utils_to_list


# ***** ==== IMPORTS TO OPTIMIZE *****
async def qualify_me(ib: IB, 
                     contracts: list,
                     desc: str = 'Qualifying contracts'):
    """[async] Qualify contracts asynchronously"""

    contracts = utils_to_list(contracts)  # to take care of single contract

    tasks = [asyncio.create_task(ib.qualifyContractsAsync(c), name=c.localSymbol) for c in contracts]

    await tqdm_asyncio.gather(*tasks, desc=desc)

    result = [r for t in tasks for r in t.result()]

    return result

def load_config():
    """Loads configuration from .env and config.yml files."""

    # Load environment variables from .env file
    load_dotenv()

    # Load config from YAML file

    ROOT = from_root()
    with open(ROOT / "config" / "config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Merge environment variables with config
    for key, value in os.environ.items():
        if key in config:
            config[key] = value

    return config


def to_list(data):
    """Converts any iterable to a list, and non-iterables to a list with a single element.

    Args:
        data: The data to be converted.

    Returns:
        A list containing the elements of the iterable, or a list with the single element if the input is not iterable.
    """

    try:
        return list(data)
    except TypeError:
        return [data]

# ***** ==== (END) IMPORTS TO OPTIMIZE *****


ROOT = from_root()
config = load_config()

# ---- SETTING CONSTANTS ----
PUTSTDMULT = config.get("PUTSTDMULT")
CALLSTDMULT = config.get("CALLSTDMULT")

PORT = port = config.get('PORT')

indexes_path = ROOT/'data'/'templates'/'snp_indexes.yml'

# --- ASSEMBLE ---------

def read_weeklys() -> pd.DataFrame:
    """gets weekly cboe symbols"""

    dls = "http://www.cboe.com/products/weeklys-options/available-weeklys"
    df = pd.read_html(dls)[0]

    return df

def rename_weekly_columns(df: pd.DataFrame) -> pd.DataFrame:
    """standardizes column names of cboe"""
    
    df.columns=['desc', 'symbol']

    return df

def remove_non_char_symbols(df: pd.DataFrame) -> pd.DataFrame:
    """removes symbols with non-chars - like dots (BRK.B)"""
    
    df = df[df.symbol.str.extract("([^a-zA-Z])").isna()[0]]

    return df

def make_weekly_cboes() -> pd.DataFrame:
    """
    Generates a weekly cboe symbols dataframe
    """

    df = (
        read_weeklys()
        .pipe(rename_weekly_columns)
        .pipe(remove_non_char_symbols)
        )
    
    # add exchange
    df = df.assign(exchange='SMART')
    
    return df


def get_snps() -> pd.Series:
    """
    gets snp symbols from wikipedia
    """
    snp_url =  "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    snps = pd.read_html(snp_url)[0]['Symbol']
    return snps

def add_snp_indexes(df: pd.DataFrame, path_to_yaml_file: str) -> pd.DataFrame:
    """
    add indexes from config/snp_indexes.yaml
    """
    with open(path_to_yaml_file, 'r') as f:
        kv_pairs = yaml.load(f, Loader=yaml.FullLoader)

    dfs = []
    for k in kv_pairs.keys():
        dfs.append(pd.DataFrame(list(kv_pairs[k].items()), 
            columns=['symbol', 'desc'])
            .assign(exchange = k))
        
    more_df = pd.concat(dfs, ignore_index=True)

    df_all = pd.concat([df, more_df], ignore_index=True)
    
    return df_all


def split_snp_stocks_and_index(df: pd.DataFrame) -> pd.DataFrame:
    """differentiates stocks and index"""
    
    df = df.assign(secType=np.where(df.desc.str.contains('Index'), 'IND', 'STK'))

    return df


def make_snp_weeklies(indexes_path: Path):
    """Makes snp weeklies with indexes"""

    # get snp stock weeklies
    df_weekly_cboes = make_weekly_cboes()
    snps = get_snps()

    # filter weekly snps
    df_weekly_snps = df_weekly_cboes[df_weekly_cboes.symbol.isin(snps)] \
                    .reset_index(drop=True)

    # add index weeklies
    df_weeklies = (
        add_snp_indexes(df_weekly_snps, indexes_path)
        .pipe(split_snp_stocks_and_index)
        )
    
    return df_weeklies


def make_unqualified_snp_underlyings(df: pd.DataFrame) -> pd.DataFrame:
    """Build underlying contracts"""

    contracts = [Stock(symbol=symbol, exchange=exchange, currency='USD') 
                if 
                    secType == 'STK'
                else 
                    Index(symbol=symbol, exchange=exchange, currency='USD') 
                for 
                    symbol, secType, exchange in zip(df.symbol, df.secType, df.exchange)]
    
    df = df.assign(contract = contracts)

    return df

async def assemble_snp_underlyings(port: int) -> dict:
    """[async] Assembles a dictionary of SNP underlying contracts"""

 
    df = make_snp_weeklies(indexes_path) \
         .pipe(make_unqualified_snp_underlyings)
    
    contracts = df.contract.to_list()

    with await IB().connectAsync(port=port) as ib:
    
        qualified_contracts = await qualify_me(ib, contracts, desc="Qualifying SNP Unds")

    return qualified_contracts


if __name__ == "__main__":

    df = asyncio.run(assemble_snp_underlyings(port))

    print(df)