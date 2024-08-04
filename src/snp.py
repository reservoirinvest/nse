# --- SNP SPECIFIC FUNCTIONS ---
# ===============================

import asyncio
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from from_root import from_root
from ib_async import IB, Contract, Index, Option, Stock
from ib_insync import IB, Contract
from loguru import logger
from tqdm.asyncio import tqdm

from ibfuncs import qualify_me
from utils import chunk_me, clean_ib_util_df, get_pickle, load_config, pickle_me, to_list

# ***** ==== (END) IMPORTS TO OPTIMIZE *****


ROOT = from_root()

PKL = ROOT / 'data' / 'zpkl'

config = load_config()

# ---- SETTING CONSTANTS ----
PUTSTDMULT = config.get("PUTSTDMULT")
CALLSTDMULT = config.get("CALLSTDMULT")

PORT = port = config.get('SNP_LIVE_PORT')

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

# --- SEEKERS ---
# ---------------




async def get_tick_data(ib: IB, c: Contract, delay: float = 0):
    """
    [async] Gets tick-by-tick data
    Quick when market is open
    Takes ~6 secs after market hours.
    No impliedVolatility
    
    Parameters:
    ib (IB): The IB instance for API interaction.
    c (Contract): The contract for which to get tick data.
    delay (float): Optional delay before returning data.
    
    Returns:
    ticker: The tick-by-tick data for the given contract.
    """

    # Request tick-by-tick data for the given contract asynchronously
    ticker = await ib.reqTickersAsync(c)
    
    # Introduce an optional delay if specified
    await asyncio.sleep(delay)

    # Return the retrieved ticker data
    return ticker

async def get_market_data(ib: IB, 
                          c: Contract,
                          sleep: float = 2):

    """
    [async] Get marketPrice including implied volatility
    Pretty quick when market is closed
    """
    tick = ib.reqMktData(c, genericTickList="106")
    try:
        await asyncio.sleep(sleep)
    finally:
        ib.cancelMktData(c)

    return tick

import logging
import math

logger = logging.getLogger(__name__)

async def get_a_price_iv(ib, contract, sleep: float=2) -> dict:
    """[async] Computes price and IV of a contract.

    OUTPUT: dict{localsymbol, price, iv}
    
    Could take up to 12 seconds in case live prices are not available"""
    
    mkt_data = await get_market_data(ib, contract, sleep)
    undPrice = mkt_data.marketPrice()

    if math.isnan(undPrice):
        undPrice = mkt_data.close
        if math.isnan(undPrice):
            tick_data = await get_tick_data(ib, contract)
            tick_data_price = tick_data[0].marketPrice()
            undPrice = tick_data_price if not math.isnan(tick_data_price) else tick_data[0].close
            if math.isnan(undPrice):
                logger.info(f"No price found for {contract.localSymbol}!")

    iv = mkt_data.impliedVolatility
    return {'localsymbol': contract.localSymbol, 'price': undPrice, 'iv': iv}

import asyncio


async def get_mkt_prices(port: int, 
                         contracts: list, 
                         chunk_size: int=44, 
                         sleep: int=7) -> pd.DataFrame:
    
    """[async] A faster way to get market prices.
    """

    contracts = to_list(contracts)
    chunks = chunk_me(contracts, chunk_size)
    results = dict()
    
    ib = await IB().connectAsync(port=port)
    try:
        for cts in tqdm(chunks, desc="Mkt prices with IVs"):
            tasks = [get_a_price_iv(ib, c, sleep) for c in cts]
            res = await asyncio.gather(*tasks)

            for r in res:
                symbol, price, iv = r
                results[symbol] = (price, iv)

        df_prices = split_symbol_price_iv(results)
        df_prices = pd.merge(clean_ib_util_df(contracts).iloc[:, :6], df_prices, on='symbol')

        # remove unnecessary columns (for secType == `STK`)
        keep_cols = ~((df_prices == 0).all() | \
                  (df_prices == "").all() | \
                    df_prices.isnull().all())

        df_prices = df_prices.loc[:, keep_cols[keep_cols == True].index]
    finally:
        await ib.disconnectAsync()

    return df_prices


def split_symbol_price_iv(prices_dict: dict) -> pd.DataFrame:
    """Splits symbol, prices and ivs into a df.
    To be used after get_mkt_prices()"""
    
    symbols, prices, ivs = zip(*((symbol, price, iv) for symbol, (price, iv) in prices_dict.items()))
    
    df_prices = pd.DataFrame({'symbol': symbols, 'price': prices, 'iv': ivs})

    return df_prices

import asyncio
from ib_async import IB

if __name__ == "__main__":

    # und_contracts = asyncio.run(assemble_snp_underlyings(port))
    # pickle_me(und_contracts, PKL)

    # und_contracts = get_pickle(PKL)

    # df_und_prices = asyncio.run(get_mkt_prices(port, und_contracts))

    # pickle_me(df_und_prices, PKL)
    # print(df_und_prices.head())

    ib = IB().connect(port=port)

    watch_dict = {"TSLA": "NYSE",
                "MSFT": "NYSE",
                "AAPL": "NYSE"}

    def wait_for_market_data(tickers):
        """print tickers as they arrive"""
        print(tickers)

    market_data={} # store ticker here
    contracts ={} # store contracts here

    # Define stocks
    for ticker in list(watch_dict.keys()):

        print(ticker)
        contracts[ticker] = Stock(ticker, watch_dict[ticker], 'USD')
        print(f"contract:{contracts[ticker]}")

        # Request current prices
        market_data[ticker] = ib.reqMktData(contracts[ticker], '', False, False)

        #ib.sleep(2)
        print(market_data[ticker])
        ib.pendingTickersEvent += wait_for_market_data

    # wait for tickers to fill
    ib.sleep(2)
    print(market_data)

    print("wait for pendingTickersEvent to produce data")
    ib.sleep(3)
    _ = [ib.cancelMktData(_c) for _c in contracts.values()]
    print("the end")
    ib.disconnect()




