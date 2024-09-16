# * --- IBKR API SPECIFIC FUNCTIONS ----
# ====================================

import asyncio
import itertools
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Union

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from from_root import from_root
from ib_async import IB, Contract, LimitOrder, MarketOrder, Option, Order, Stock, util
from loguru import logger
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from utils import (chunk_me, clean_ib_util_df, convert_to_utc_datetime, empty_the_df, get_dte, get_port, load_config, pickle_me, split_symbol_price_iv, to_list)

ROOT = from_root()
dotenv_path = find_dotenv()
load_dotenv(dotenv_path=dotenv_path)  # loads environment from .env

LOGLEVEL = os.getenv("LOGLEVEL", "DEBUG")
ACTIVESTATUS = os.getenv("ACTIVESTATUS", "")

# * --- SETTING LOGS ----

# Set ib_async logs to file, for loguru to capture
level = logging.getLevelNamesMapping().get(LOGLEVEL)
log_file = ROOT / "log" / str(__name__ + ".log")
util.logToFile(log_file, level=level)
open(log_file, "w").close()  # Wipe the logfile clean!

# * --- CLASSES AND THEIR METHODS

@dataclass
class OpenOrder:
    """
    Open order template with Dummy data. Use:\n
    `df = OpenOrder().empty()`
    """

    conId: int = 0
    symbol: str = "Dummy"
    secType: str = "STK"
    expiry: datetime = datetime.now()
    strike: float = 0.0
    right: str = "?"  # Will be 'P' for Put, 'C' for Call
    orderId: int = 0
    order: Order = None
    permId: int = 0
    action: str = "SELL"  # 'BUY' | 'SELL'
    totalQuantity: float = 0.0
    lmtPrice: float = 0.0
    status: str = None

    def empty(self):
        return empty_the_df(self)


@dataclass
class Portfolio:
    """
    Portfolio template with Dummy data. Use:\n
    `df = OpenOrder().empty()`
    """

    conId: int = 0
    symbol: str = "Dummy"
    secType: str = "STK"
    expiry: datetime = datetime.now()
    strike: float = 0.0
    right: str = "?"  # Will be 'P' for Put, 'C' for Call
    position: float = 0.0
    mktPrice: float = 0.0
    mktVal: float = 0.0
    avgCost: float = 0.0
    unPnL: float = 0.0
    rePnL: float = 0.0

    def empty(self):
        return empty_the_df(self)

# *--- MASS PROCESS IN CHUNKS ----

async def process_in_chunks(ib: IB,
                            data: any,
                            func: callable = None,
                            func_args: dict = None,
                            chunk_size: int = 25,
                            chunk_desc: str = "processing chunk...",
) -> list:
    """Processes functions in chunks.

    Args:
        ib (IB): A live connection
        data (any): Single contract | list | pd.Series of contracts
        func (callable, optional): The function to be processed. Defaults to None.
        func_args (dict, optional): Arguments supplied to the function. Defaults to None.
        chunk_size (int, optional): Size of processing. Defaults to 25.
        chunk_desc (str, optional): Description while processing. Defaults to "Processing chunk".

    Raises:
        ValueError: _description_
        TypeError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if not func:
        raise ValueError("A function must be provided for processing the data.")
        return None

    if not func_args:
        func_args = {}

    chunks = chunk_me(data, chunk_size)
    processed_data = []

    for chunk in tqdm(chunks, desc=chunk_desc):
        func_args["data"] = chunk

        try:
            # Attempt to process the chunk with the function
            # print('First attempt...')
            result = await func(ib, **func_args)
            # print(result) # !!! TEMPORARY
            processed_chunk = [result]

        except (AttributeError, ValueError) as e:

            # If an error occurs, check if the function accepts unpacked data
            try:
                # Unpack the chunk and call the function again
                # print(f'Second attempt due to {e}: {type(e)}')

                if isinstance(chunk, (list, pd.Series)):
                    tasks = []
                    farg2 = func_args
                    for item in chunk:
                        farg2["data"] = item
                        t = [func(ib, **farg2)]
                        tasks.extend(t)
                    result = await asyncio.gather(*tasks)
                    processed_chunk = [clean_ib_util_df(chunk).join(pd.DataFrame(result)).drop(columns='localsymbol')]

                else:
                    # If the chunk is not iterable, raise an error
                    raise TypeError(f"Invalid data type: {type(chunk)} for function {func.__name__}")

            except Exception as e2:
                # Raise a custom error if it still fails
                raise ValueError(f"Function {func.__name__} does not accept data type {type(chunk)}, \nerror:{e2}") from e2

        processed_data.extend(processed_chunk)

    return processed_data


# *---- QUALIFYING ----

async def qualify_me(ib: IB, data: list, desc: str = "Qualifying contracts") -> list:
    """[async] Qualify contracts asynchronously"""

    data = to_list(data)  # to take care of single contract

    tasks = [
        asyncio.create_task(ib.qualifyContractsAsync(c), name=c.localSymbol)
        for c in data
    ]

    await tqdm_asyncio.gather(*tasks, desc=desc)

    result = [r for t in tasks for r in t.result()]

    return result


# *--- SEEKING ---

def get_ib(MARKET: str, cid: int = 10, LIVE: bool = True) -> IB:
    """Gets an active IB port for context managers

    Args:
        MARKET (str): NSE | SNP
        cid (int, optional): clientId for IB. Defaults to 10.
        LIVE (bool, optional): LIVE or PAPER

    Returns:
        IB: an active IB connection
    """
    port = get_port(MARKET=MARKET, LIVE=LIVE)

    connection = IB().connect(port=port, clientId=cid)

    return connection


def quick_pf(ib: IB) -> Union[None, pd.DataFrame]:
    """Gets the portfolio dataframe"""
    pf = ib.portfolio()  # returns an empty [] if there is nothing in the portfolio

    if pf != []:
        df_pf = util.df(pf)
        df_pf = (util.df(list(df_pf.contract)).iloc[:, :6]).join(
            df_pf.drop(columns=["account"])
        )
        df_pf = df_pf.rename(
            columns={
                "lastTradeDateOrContractMonth": "expiry",
                "marketPrice": "mktPrice",
                "marketValue": "mktVal",
                "averageCost": "avgCost",
                "unrealizedPNL": "unPnL",
                "realizedPNL": "rePnL",
            }
        )
    else:
        df_pf = Portfolio().empty()

    return df_pf


async def account_values(ib: IB) -> dict:
    """Gets account values

    Args:
        ib (IB): an active connection

    Returns:
        dict: current nlv, cash and margins
    """

    df_acc = util.df(ib.accountValues())

    d_map = {
        "TotalCashBalance": "cash",
        "Cushion": "cushion",
        "NetLiquidation": "nlv",
        "InitMarginReq": "init_margin",
        "EquityWithLoanValue": "equity_val",
        "MaintMarginReq": "maint_margin",
        "realizedPnL": "pnl_real",
        "UnrealizedPnL": "pnl_unreal",
        "LookAheadAvailableFunds": "funds_avlbl",
    }

    # get account values as a dictionary
    df_out = df_acc[df_acc.tag.isin(d_map.keys())]
    acc = df_out.set_index("tag").value.apply(float).to_dict()

    # sort account values based on d_map's order
    order = list(d_map.values())
    order_index = {key: index for index, key in enumerate(order)}
    sorted_keys = sorted(d_map.keys(), key=lambda x: order_index.get(x, float("inf")))
    sorted_dict = {d_map.get(key): acc.get(key) for key in sorted_keys}

    return sorted_dict

def get_nlv(MARKET: str, save: bool=True) -> dict:
    """Gets net liquidation value, pnl, cash and cushion

    Args:
        MARKET (str): SNP | NSE
        save (bool, optional): pickles the dict. Defaults to True.

    Returns:
        dict: of cash, cushion, margins and pnl
    """

    with get_ib(MARKET) as ib:
        nlv = ib.run(account_values(ib))


    if save:
        ROOT = from_root()
        pickle_me(nlv, ROOT/'data'/str(MARKET.lower()+'_nlv.pkl'))

    return nlv


def get_pf_margins(MARKET: str, save:bool=True) -> pd.DataFrame:
    """Generates portfolios with margins and volatilities

    Args:
        MARKET (str): SNP | NSE
        save (bool, optional): pickles if true. Defaults to True.

    Returns:
        pd.DataFrame: portfolio with und price, iv and hv
    """

    with get_ib(MARKET) as ib:

        df_pf = quick_pf(ib)

        # prepare the wif orders
        df_pf = df_pf.assign(action=df_pf.position.apply(lambda x:
                                            'BUY' if x < 0
                                            else('SELL' if x > 0 else None)))

        df_pfm = df_pf.assign(order=[MarketOrder(action, qty)
                            for action, qty
                            in zip(df_pf.action, df_pf.position.abs())])

        pfc = [Contract(conId=c) for c in df_pf.conId]
        pfc = ib.run(qualify_me(ib, pfc, desc='Qualifying portfolios'))
        df_pfm = df_pfm.assign(contract=pfc)

        df_mcom = ib.run(marginsAsync(ib=ib, data=df_pfm, timeout=5))
        df_mcom.comm = 20 if MARKET == 'NSE' else df_mcom.comm
        dfm = df_pfm.drop(columns=['order', 'action']).join(df_mcom.drop(columns='contract'))
        dfm.rename(columns={'symbol': 'ib_symbol'}, inplace=True, errors='ignore')

        # ... get exchange and currency from a contract
        c = pfc[0]

        if any(e =='' for e in {c.exchange for c in pfc}):
            exchange = c.primaryExchange
        else:
            exchange = c.exchange

        currency = c.currency

        # ... get market prices for underlyings
        symbols = set(c.symbol for c in pfc)

        unds =[Stock(s, exchange, currency) for s in symbols]
        unds = ib.run(qualify_me(ib, unds, desc="Qualifying underlyings"))

        dfmp = ib.run(get_mkt_prices(ib, unds))

    # ... integrate df_unds
    df_unds = dfmp[['ib_symbol', 'price', 'iv', 'hv']].rename(columns={'price': 'undPrice', 'iv': 'und_iv', 'hv': 'und_hv'})
    dfu = dfm.set_index('ib_symbol').join(df_unds.set_index('ib_symbol')).reset_index().drop(columns=[c for c in dfm.columns])

    insert_pos = 5
    df = pd.concat([dfm.iloc[:, :insert_pos], dfu, dfm.iloc[:, insert_pos:]], axis=1)

    df = df.assign(
    expiry=df.expiry.apply(lambda x:
                           convert_to_utc_datetime(x, eod=True)))

    df.insert(4, 'dte', get_dte(df.expiry))

    if save:
        ROOT = from_root()
        pickle_me(df, ROOT/'data'/str(MARKET.lower()+'_pf.pkl'))

    return df


# *--- Option Chains -----

async def get_an_option_chain(ib: IB, contract:Contract, timeout: int=2):
    """Gets a single option chain for a single contract with timeout

    Args:
        ib (IB): Live IB connection
        contract (Contract): an underlying contract
        timeout (int, optional): Time to generate an option chain. Defaults to 2.

    Returns:
        named tuple: an option chain
    """
    try:
        chain = await asyncio.wait_for(ib.reqSecDefOptParamsAsync(
        underlyingSymbol=contract.symbol,
        futFopExchange="",
        underlyingSecType=contract.secType,
        underlyingConId=contract.conId,
        ), timeout=timeout)

        if chain:
            chain = chain[-1] if isinstance(chain, list) else chain
        return chain
    except asyncio.TimeoutError:
        logging.error(f"Timeout occurred while getting option chain for {contract.symbol}")
        return None

async def get_option_chains(ib: IB,
                            contracts: list,
                            msg: str,
                            chunk_size:int=20,
                            timeout:float=4,
                            ) -> list:
    """Gets a list of option chains

    Args:
        ib (IB): Live IB connection
        contracts (list): List of contraacts to get option chain for
        chunk_size (int, optional): chunks of contracts. Defaults to 20.
        timeout (float, optional): timeout for an option chain. Defaults to 4.

    Returns:
        list: option chains list
    """
    option_chains = []
    total_contracts = len(contracts)

    with tqdm(total=total_contracts, unit="contract", desc=msg) as pbar:

        for i in range(0, total_contracts, chunk_size):
            chunk = contracts[i: i+chunk_size]
            tasks = [get_an_option_chain(ib, contract, timeout) for contract in chunk]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            option_chains.extend([chain for chain in results])
            pbar.update(len(chunk))

    return option_chains

def make_chains(df_unds: pd.DataFrame,
                timeout: float=15,
                chunks: int=15,
                save: bool=False,
                msg: str='Getting chains',
                ) -> pd.DataFrame:
    """makes option chains from ib

    Args:
        df_unds (pd.DataFrame): underlying df for a market
        timeout (float, optional): time to fill. Defaults to 15.
        chunks (int, optional): chunk size to process. Defaults to 15.
        save (bool, optional): to pickle. Defaults to False.

    Returns:
        pd.DataFrame: option chains
    """
    MARKET = 'NSE' if df_unds.contract.iloc[0].exchange=='NSE' else 'SNP'
    und_contracts = df_unds.contract.to_list()

    with get_ib(MARKET) as ib:
        chains = ib.run(get_option_chains(ib, und_contracts, timeout=timeout, chunk_size=chunks, msg=msg))

    # Clean the chains
    df_chains = util.df([c for c in chains if c is not None])
    df_chains.rename(columns={'underlyingConId': 'undId'}, inplace=True, errors="ignore")
    dfc = df_chains[['undId', 'expirations', 'strikes']]
    expirations = dfc.expirations.apply(lambda d: [convert_to_utc_datetime(val, eod=True) for val in d])
    dfc.loc[:, 'expirations'] = expirations

    # Create a list of tuples, each containing (undId, expiration, strike)
    data = []
    for i, row in dfc.iterrows():
        data.extend([(int(row['undId']), exp, strike, get_dte(exp), right) for exp, strike, right in itertools.product(row['expirations'], row['strikes'], ['P', 'C'])])

    # Create the new DataFrame
    df_chains = pd.DataFrame(data, columns=['undId', 'expiry', 'strike', 'dte', 'right'])
    df_chains = df_chains[df_chains.dte > 0] # remove negative dte chains

    # Join the expanded dataframe with df_unds
    dfch = pd.merge(df_unds.drop(columns=['expiry', 'strike', 'right', 'contract']), df_chains, on=['undId'], how='left')
    dfch = dfch.rename(columns={'price': 'undPrice', 'iv': 'und_iv', 'hv': 'und_hv'}, errors='ignore')

    if save:
        ROOT = from_root()
        pickle_me(dfch, ROOT/'data'/str(MARKET.lower()+'_opts.pkl'))

    return dfch

# *---- Margins and commissions -----

async def get_one_margin(ib: IB, data: Contract|Option|Stock, order, timeout: int=7) -> dict|None:
    """Get margin with commissions within a time

    Args:
        ib (IB): instance of IB class
        data (Contract | Option | Stock): a qualified contract
        order: an order
        timeout (int): delay. Defaults to 7.

    Returns:
        dict|None: _description_
    """

    try:
        wif = await asyncio.wait_for(
            ib.whatIfOrderAsync(data, order), timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.error(f"{data.localSymbol} wif timed out!")
        wif = None
    return wif


def margin_comm(r) -> dict:
    """Clean a result"""

    if r:
        margin = float(r.maintMarginChange)
        comm = min(float(r.commission), float(r.minCommission), float(r.maxCommission))
        if comm > 1e7:
            comm = np.nan
    else:
        margin = comm = np.nan

    return (margin, comm)


async def marginsAsync(
    ib: IB, data: pd.DataFrame,
    timeout: float = 2,
    ) -> pd.DataFrame:
    """Gets async contracts from a df

    Args:
        ib (IB): An active IB connection
        df (pd.DataFrame): df with `contract` and `order` fields
        timeout (float, optional): time delay to get a margin. Defaults to 2.
        eod (bool, optional): gets end of day time for options. Defaults to True.
        ist (bool, optional): gets Indian Std Time for NSE options. Defaults to True.

    Returns:
        pd.DataFrame: df with ib_symbol, margin and comm
    """
    try:
        contracts = data.contract.to_list()
        orders = data.order.to_list()
    except ValueError as e:
        logging.error(f"df does not have contract or order.Error: {e}")
        return pd.DataFrame([])

    # qualify contracts if there is no conId
    if data.contract.iloc[0].conId == 0:
        await ib.qualifyContractsAsync(*contracts)

    cos = zip(contracts, orders)

    tasks = [asyncio.create_task(get_one_margin(ib, c, o, timeout)) for c, o in cos]

    results = await asyncio.gather(*tasks)

    mcom = [margin_comm(r) for r in results]

    df1 = pd.DataFrame(mcom, columns=["margin", "comm"])
    df_mcom = df1.assign(contract=contracts)

    return df_mcom


# *---- Price and IVs ---

async def get_mkt_prices(ib:IB, data: list,
                         chunk_size: int = 44, sleep: int = 7,
                         gentick:str='106, 104') -> pd.DataFrame:
    """Gets market prices with iv and hv.

    Args:
        ib (IB): Live IB connection
        contracts (list): list of IB contracts
        chunk_size (int, optional): block size for processing. Defaults to 44.
        sleep (int, optional): delay to fill. Defaults to 7.
        gentick (str, optional): iv:106 | hv:104. Defaults to '106, 104'.

    Returns:
        pd.DataFrame: _description_
    """

    data = to_list(data)
    chunks = chunk_me(data, chunk_size)
    results = dict()

    for cts in tqdm(chunks, desc="Mkt prices with IVs"):
        tasks = [get_a_price_iv(ib, c, sleep, gentick) for c in cts]
        res = await asyncio.gather(*tasks)


        for r in res:
            symbol, price, iv, hv = r.values()
            results[symbol] = (price, iv, hv)

    df_prices = split_symbol_price_iv(results)

    dfp = clean_ib_util_df(data).join(df_prices.drop(columns='ib_symbol'))

    return dfp


async def get_tick_data(ib: IB, c: Contract, delay: float = 6):
    """Gets tick-by-tick data

    Args:
        ib (IB): IB instance
        c (Contract): a contract
        delay (float, optional): delay to fill. Defaults to 6 secs.

    Returns:
        _type_: IB ticker
    """

    # Request tick-by-tick data for the given contract asynchronously
    ticker = await ib.reqTickersAsync(c)

    # Introduce an optional delay if specified
    await asyncio.sleep(delay)

    # Return the retrieved ticker data
    return ticker


async def get_market_data(ib: IB, c: Contract,
                          sleep: float = 2,
                          gentick: str='106, 104'):
    """For a contract, gets market price with implied volatility.

    Args:
        ib (IB): Live IB connection
        c (Contract): An IB contract
        sleep (float, optional): delay to fill. Defaults to 2.
        gentick (str, optional): iv:106 | hv:104. Defaults to '106, 104'.

    Returns:
        _type_: _description_
    """

    tick = ib.reqMktData(c, genericTickList=gentick)
    try:
        await asyncio.sleep(sleep)
    finally:
        ib.cancelMktData(c)

    return tick


async def get_a_price_iv(ib:IB, data: Contract|Option|Stock, sleep: float = 15, gentick: str='106, 104') -> dict:
    """Computes price and ivs of a contract. Picks `close` price in closed market.

    Args:
        ib (IB): Active IB connection
        data (Contract | Option | Stock): an IB contract
        sleep (float, optional): Defaults to 15.
        gentick (str, optional): iv:106 | hv:104. Defaults to '106'.

    Returns:
        dict: {'localsymbol': str, 'iv': float, 'hv':float}
    """

    mkt_data = await get_market_data(ib, data, sleep, gentick)
    data = mkt_data.__dict__

    price_dict = {k: v for k, v in data.items() if k in ['close', 'last']}
    localSymbol = data.get('contract').localSymbol

    undPrice = price_dict.get('last') if not pd.isnull(price_dict.get('last')) else price_dict.get('close')
    iv = data.get('impliedVolatility')
    hv = data.get('histVolatility')

    if math.isnan(undPrice):
        logger.info(f"No price found for {localSymbol}!")

    return {"localsymbol": localSymbol, "price": undPrice, "iv": iv, "hv": hv}


# * --- ORDER HANDLING ---

def order_nakeds():
    """ # !!! To be made from _order_nse.ipynb!!!"""
    pass


def make_ib_orders(df: pd.DataFrame) -> tuple:
    """Make (contract, order) tuples"""

    contracts = df.contract.to_list()
    orders = [
        LimitOrder(action="SELL", totalQuantity=abs(int(q)), lmtPrice=p)
        for q, p in zip(df.lot, df.xPrice)
    ]

    cos = tuple((c, o) for c, o in zip(contracts, orders))

    return cos


def place_orders(ib: IB, cos: Union[tuple, list], blk_size: int = 25) -> List:
    """!!!CAUTION!!!: This places orders in the system
    ---
    NOTE: cos could be a single (contract, order)
          or a tuple/list of ((c1, o1), (c2, o2)...)
          made using tuple(zip(cts, ords))
    ---
    USAGE:
    ---
    cos = tuple((c, o) for c, o in zip(contracts, orders))
    with IB().connect(port=port) as ib:
        ordered = place_orders(ib=ib, cos=cos)
    """

    trades = []

    if isinstance(cos, (tuple, list)) and (len(cos) == 2):
        c, o = cos
        trades.append(ib.placeOrder(c, o))

    else:
        cobs = {cos[i : i + blk_size] for i in range(0, len(cos), blk_size)}

        for b in tqdm(cobs):
            for c, o in b:
                td = ib.placeOrder(c, o)
                trades.append(td)
            ib.sleep(0.75)

    return trades


def cancel_orders(ib: IB, orders: list, blk_size: int = 25) -> list:
    """Cancels orders if they exist

    Args:
        ib (IB): a live IB connection
        orders (list): list of orders to be cancelled
        blk_size (int, optional): cancel in chunks. Defaults to 25.

    Returns:
        list: list of cancelled orders
    """

    cancels = []
    order_blks = [orders[i : i + blk_size] for i in range(0, len(orders), blk_size)]
    for b in tqdm(order_blks):
        for o in b:
            td = ib.cancelOrder(o)
            cancels.append(td)
        ib.sleep(0.75)

    return cancels


def cancel_all_orders(ib: IB) -> list:
    """Cancels all Open Orders

    Args:
        ib (IB): an active IB connection

    Returns:
        list: list of cancelled open orders
    """

    df_ords = get_open_orders(ib)
    orders = df_ords.order.to_list()

    cancels = cancel_orders(ib, orders)

    logger.info(f"Cancelled {len(orders)} orders")

    return cancels

def get_open_orders(ib, is_active: bool = False) -> pd.DataFrame:
    """Gets open orders - blocking version"""

    df_openords = OpenOrder().empty()  # Initialize open orders

    trades = ib.reqAllOpenOrders()

    dfo = pd.DataFrame([])

    if trades:
        all_trades_df = (
            clean_ib_util_df([t.contract for t in trades])
            .join(util.df(t.orderStatus for t in trades))
            .join(util.df(t.order for t in trades), lsuffix="_")
        )

        order = pd.Series([t.order for t in trades], name="order")

        all_trades_df = all_trades_df.assign(order=order)

        all_trades_df.rename(
            {"lastTradeDateOrContractMonth": "expiry",
             "symbol": "ib_symbol"}, axis="columns", inplace=True
        )

        # print("Available columns:", all_trades_df.columns)  # Debug print
        # print("trades_cols:", df_openords.columns)  # Debug print

        # Check if 'symbol' is in the DataFrame, if not, try to find an alternative
        if 'symbol' not in all_trades_df.columns:
            if 'contract' in all_trades_df.columns:
                all_trades_df['symbol'] = all_trades_df['contract'].apply(lambda x: x.symbol)
            else:
                raise ValueError("Neither 'symbol' nor 'contract' column found in the DataFrame")

        dfo = all_trades_df[df_openords.columns]

        if is_active:
            dfo = dfo[dfo.status.isin(ACTIVESTATUS)]

    return dfo


if __name__ == "__main__":
    MARKET = "nse"
    config = load_config(MARKET)

    port = config.get("PORT")

    with IB().connect(port=port, clientId=10) as ib:
        out = quick_pf(ib)

    print(out)
