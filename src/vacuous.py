# ---- UNNECESSARY CODE NOT CONTRIBUTING ANYTHING----


import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger
from tqdm import tqdm

from nse import NSEfnos, equity_iv_df


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
