# * --- IBKR API SPECIFIC FUNCTIONS ----
# ====================================

import asyncio
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
from ib_async import IB, Contract, LimitOrder, MarketOrder, Option, Order, util
from loguru import logger
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from utils import (arrange_orders, chunk_me, clean_ib_util_df, get_port, handle_nse_raws,
                   load_config, make_contracts_orders, pickle_me, split_symbol_price_iv, to_list)

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

# --- CLASSES AND THEIR METHODS

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


def empty_the_df(df):
    """Empty the dataclass df"""
    empty_df = pd.DataFrame([df.__dict__]).iloc[0:0]
    return empty_df


# *--- IB BLOCKING FUNCTIONS ---


def get_ib_margin(contract: Option, order: MarketOrder, port: int) -> dict:
    """Gets margin and commission of a contract"""

    with IB().connect(port=port) as ib:
        if contract.conId == 0:  # qualify raw contracts
            contract = next(iter(ib.qualifyContracts(contract)))
        wif = ib.whatIfOrder(contract, order)

    # margin = float(wif.initMarginChange) # initial margin is too high compared to Zerodha, SAMCO
    margin = float(wif.maintMarginChange)
    comm = min(
        float(wif.commission), float(wif.minCommission), float(wif.maxCommission)
    )
    if comm > 1e7:
        comm = np.nan

    return {"contract": contract, "margin": margin, "comm": comm}


def get_ib_margin_comms(df: pd.DataFrame, port: int) -> pd.DataFrame:
    """Qualified Contracts, Margins and Commissions from an options df"""

    symbol = df.ib_symbol.iloc[0]
    df_cos = make_contracts_orders(df)

    cts = [d if d.conId == 0 else None for d in df_cos.contract]
    with IB().connect(port=port) as ib:
        if len(cts) > 40:
            ib.qualifyContracts(*tqdm(cts, desc=f"Qualifying {symbol} options"))
        else:
            ib.qualifyContracts(*cts)

        df_cos.contract = cts
        ib.disconnect()

    if len(df_cos) > 1:  # use tqdm.pandas.progress_apply()
        tqdm.pandas(desc=f"Calculating {symbol} margins")
        data = df_cos.progress_apply(
            lambda row: get_ib_margin(row.contract, row.order, port=port), axis=1
        )
    else:
        data = df_cos.apply(
            lambda row: get_ib_margin(row.contract, row.order, port=port), axis=1
        )

    df_mcom = pd.DataFrame.from_dict(data.to_dict()).T

    # replace raw contracts with qualified
    df_q = df_cos.join(df_mcom, how="outer", lsuffix="_left").drop(
        ["contract_left", "order"], axis=1
    )

    # merge margins and commissions
    df_opts = df.merge(df_q, left_index=True, right_index=True, suffixes=("_left", ""))
    df_opts = df_opts.drop(columns="contract_left", errors="ignore")

    # determine the secType for IB
    df_opts = df_opts.assign(secType=df_opts.contract.apply(lambda s: s.secType))

    return df_opts


# *---- QUALIFYING ----

async def qualify_me(ib: IB, contracts: list, desc: str = "Qualifying contracts"):
    """[async] Qualify contracts asynchronously"""

    contracts = to_list(contracts)  # to take care of single contract

    tasks = [
        asyncio.create_task(ib.qualifyContractsAsync(c), name=c.localSymbol)
        for c in contracts
    ]

    await tqdm_asyncio.gather(*tasks, desc=desc)

    result = [r for t in tasks for r in t.result()]

    return result



# *--- SEEKING ---

def get_ib(MARKET:str, cid: int=10) -> IB:
    """Gets an active IB port for context managers

    Args:
        MARKET (str): NSE | SNP
        cid (int, optional): clientId for IB. Defaults to 10.

    Returns:
        IB: an active IB connection
    """

    port = get_port(MARKET)
    return IB().connect(port=port, clientId=cid)


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


# *---- Margins and commissions -----


async def get_one_margin(ib, contract, order, timeout):
    """Get margin with commissions within a time"""

    try:
        wif = await asyncio.wait_for(
            ib.whatIfOrderAsync(contract, order), timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.error(f"{contract.localSymbol} wif timed out!")
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
    ib: IB, df: pd.DataFrame, timeout: float = 2, eod: bool = True, ist: bool = True
) -> pd.DataFrame:
    """Gets async contracts from a df
    Args:
      df: dataframe with `contract` and `order` columns
      port: ib port
      timeout: time to wait. ~2 seconds for 10 rows
    Returns:
      a Dataframe with same index as input"""

    try:
        contracts = df.contract.to_list()
        orders = df.order.to_list()
    except ValueError as e:
        logging.error(f"df does not have contract or order.Error: {e}")
        return pd.DataFrame([])

    # qualify contracts if there is no conId
    if df.contract.iloc[0].conId == 0:
        await ib.qualifyContractsAsync(*contracts)

    cos = zip(contracts, orders)

    tasks = [asyncio.create_task(get_one_margin(ib, c, o, timeout)) for c, o in cos]

    results = await asyncio.gather(*tasks)

    mcom = [margin_comm(r) for r in results]

    df1 = pd.DataFrame(mcom, columns=["margin", "comm"])
    df_mcom = df1.assign(contract=contracts)

    return df_mcom


# *---- Price and IVs ---

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


async def get_market_data(ib: IB, c: Contract, sleep: float = 2):

    """Gets market price with implied volatility. Works also in closed market.

    Args:
        ib (IB): IB instance
        c (Contract): a contract
        sleep (float, optional): delay to fill. Defaults to 2 secs.

    Returns:
        _type_: IB tick_
    """

    tick = ib.reqMktData(c, genericTickList="106")
    try:
        await asyncio.sleep(sleep)
    finally:
        ib.cancelMktData(c)

    return tick


async def get_a_price_iv(ib, contract, sleep: float = 2) -> dict:
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
            undPrice = (
                tick_data_price
                if not math.isnan(tick_data_price)
                else tick_data[0].close
            )
            if math.isnan(undPrice):
                logger.info(f"No price found for {contract.localSymbol}!")

    iv = mkt_data.impliedVolatility
    return {"localsymbol": contract.localSymbol, "price": undPrice, "iv": iv}


async def get_mkt_prices(
   ib:IB, contracts: list, chunk_size: int = 44, sleep: int = 7
) -> pd.DataFrame:
    """[async] A faster way to get market prices."""

    contracts = to_list(contracts)
    chunks = chunk_me(contracts, chunk_size)
    results = dict()

    for cts in tqdm(chunks, desc="Mkt prices with IVs"):
        tasks = [get_a_price_iv(ib, c, sleep) for c in cts]
        res = await asyncio.gather(*tasks)

        for r in res:
            symbol, price, iv = r.values()
            results[symbol] = (price, iv)

    df_prices = split_symbol_price_iv(results)
    df_prices = pd.merge(
        clean_ib_util_df(contracts), df_prices, on="symbol"
    )

    return df_prices

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
            {"lastTradeDateOrContractMonth": "expiry"}, axis="columns", inplace=True
        )

        trades_cols = df_openords.columns

        dfo = all_trades_df[trades_cols]

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
