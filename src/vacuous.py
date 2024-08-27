# ---- UNNECESSARY CODE NOT CONTRIBUTING ANYTHING----

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger
from tqdm import tqdm
from ib_async import IB, MarketOrder, Option

from ibfuncs import qualify_me
from nse import NSEfnos, equity_iv_df
from utils import chunk_me, make_contracts_orders


def make_raw_fno_df(fnos) -> pd.DataFrame:
    """Makes all the raw fnos"""

    n = NSEfnos()

    dfs = []

    with tqdm(total=len(fnos), desc="Generating raw fnos", unit="symbol") as pbar:
        for s in fnos:
            try:
                df = equity_iv_df(n.stock_quote_fno(s))
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error for {s} - error: {e}")
                pass

            pbar.update(1)

    df = pd.concat(dfs, ignore_index=True)

    return df


def rbi_tr_to_json(wrapper):
    trs = wrapper.find_all("tr")
    op = {}
    for tr in trs:
        tds = tr.find_all("td")
        if len(tds) >= 2:
            key = tds[0].text.strip()
            val = tds[1].text.replace(":", "").replace("*", "").replace("#", "").strip()

            op[key] = val
    return op


class RBI:
    """Not working due to captcha"""

    base_url = "https://www.rbi.org.in/"

    def __init__(self):
        self.s = requests.Session()

    def current_rates(self):
        r = self.s.get(self.base_url)

        bs = BeautifulSoup(r.text, "html.parser")
        wrapper = bs.find("div", {"id": "wrapper"})

        return rbi_tr_to_json(wrapper)

    def repo_rate(self):
        rate = self.current_rates().get("Policy Repo Rate")[:-1]

        return float(rate)



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


async def qualify_in_chunks(ib: IB, contracts: list, chunk_size: int = 200, desc: str = "qualifying chunk"):
    """
    Qualify a list of contracts in chunks using the `qualify_me()` function.

    Args:
        ib (IB): An instance of the IB class.
        contracts (list): A list of contracts to be qualified.
        chunk_size (int, optional): The size of each chunk. Defaults to 200.
        desc (str, optional): The description to be used in the tqdm progress bar. Defaults to "Qualifying contracts".

    Returns:
        list: The qualified contracts.
    """
    chunks = chunk_me(contracts, chunk_size)
    qualified_contracts = []

    for chunk in tqdm(chunks, desc=desc):
        qualified_chunk = await qualify_me(ib, chunk)
        qualified_contracts.extend(qualified_chunk)

    return qualified_contracts
