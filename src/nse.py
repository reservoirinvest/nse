# --- NSE-SPECIFIC FUNCTIONS ---
# ==============================

import asyncio
import io
import json
import math
from datetime import datetime, timedelta, timezone
from typing import List, Union

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from from_root import from_root
from ib_async import IB
from loguru import logger
from pandas import json_normalize
from tqdm import tqdm

from ibfuncs import get_ib_margin_comms, marginsAsync
from utils import (Timer, append_black_scholes, append_cos,
                   append_safe_strikes, append_xPrice, black_scholes,
                   convert_to_numeric, convert_to_utc_datetime, get_dte,
                   get_pickle_suffix, get_prec, load_config,
                   merge_and_overwrite_df, pickle_me, split_dates)

ROOT = from_root()
config = load_config()

# ---- SETTING CONSTANTS ----

# maps  for nifty and bank nifty
IDXHISTSYMMAP = config.get("IDXHISTSYMMAP")
PUTSTDMULT = config.get("PUTSTDMULT")
CALLSTDMULT = config.get("CALLSTDMULT")

PORT = port = config.get('PORT')


# ------ NSE FUNCTIONS ----

def make_earliest_nakeds(fnos: Union[List, set], 
                         save: bool = False) -> pd.DataFrame:
    """Make all early fnos"""

    timer = Timer("Making earliest nakeds")
    timer.start()

    dfs = []

    with tqdm(total=len(fnos), desc="Making nakeds", unit="symbol") as pbar:

        for symbol in fnos:

            pbar.set_description(f"for: {symbol}")

            try:
                df_nakeds = make_early_opts_for_symbol(symbol, port=port)
                df_nakeds = df_nakeds[df_nakeds.xPrice > 0]

            except (AttributeError, ValueError) as e:
                logger.error(e)
                df_nakeds = None

            dfs.append(df_nakeds)

            # collect dfs and save
            if dfs:
                try:
                    df = pd.concat(dfs, axis=0, ignore_index=True)
                except ValueError as e:
                    logger.error(f"No dfs to concat!. Error: {e}")
                    df = pd.DataFrame([])

                # sort by likeliest
                if not df.empty:
                    df = df.loc[(df.xPrice / df.price).sort_values().index]

                if save and not df.empty:
                    suffix = get_pickle_suffix(pattern="*nakeds*")
                    filename = str(f"earliest_nakeds{suffix}.pkl")
                    pickle_me(df, ROOT / "data" / "raw" / filename)
            
            else:
                df = dfs

            pbar.update(1)

    timer.stop()

    return df


def make_early_opts_for_symbol(
    symbol: str,  # nse_symbol. can be equity or index
    port: int,
    timeout: int=2, # timeout for marginsAsync
        ) -> pd.DataFrame:
    """Make target options for nakeds with earliest expiry
    
    Args:
       symbol: can be equity or index
       port: int. Could be LIVE_PORT | PAPER_PORT
       """

    # initialize and get the base df
    n = NSEfnos()
    q = n.stock_quote_fno(symbol)
    dfe = equity_iv_df(q)
    
    # clean up zero IVs and dtes
    mask = (dfe.iv > 0) & (dfe.dte > 0)
    df = dfe[mask].reset_index(drop=True)
    
    # Append safe strikes
    df = append_safe_strikes(df)
    
    # Append black scholes
    
    rbi = RBI()
    risk_free_rate = rbi.repo_rate() / 100
    
    df = append_black_scholes(df, risk_free_rate)
    
    # Append contract, order to prep for margin
    df = append_cos(df)
    
    # Get margins with approrpriate timeout and append
    with IB().connect(port=port) as ib:
        df_mcom = ib.run(marginsAsync(ib=ib, df=df, 
                                           timeout=timeout))
    
    df = merge_and_overwrite_df(df, df_mcom)
    
    # Append xPrice
    df = append_xPrice(df)

    return df


def nse_ban_list() -> list:
    """Gets scrips banned today"""

    url = "https://nsearchives.nseindia.com/content/fo/fo_secban.csv"
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/80.0.3987.149 Safari/537.36",
        "accept-language": "en,gu;q=0.9,hi;q=0.8",
        "accept-encoding": "gzip, deflate, br",
    }

    base_url = "https://www.nseindia.com"

    with requests.Session() as session:
        request = session.get(base_url, headers=headers, timeout=5)
        cookies = dict(request.cookies)
        response = session.get(url, headers=headers, timeout=5, cookies=cookies)

    df = pd.read_csv(io.StringIO(response.text))
    ban_list = df.iloc[:, 0].tolist()

    return ban_list


# --- SEEKING ---

def get_all_fno_names() -> set:
    """All fnos in nse, including index"""

    n = NSEfnos()
    d = n.stock_quote_fno('NIFTY')
    fnos = n.equities() | set(d.get('allSymbol'))
    return fnos


def make_raw_fno_df(fnos) -> pd.DataFrame:
    """Makes all the raw fnos"""

    n = NSEfnos()

    dfs = []

    with tqdm(total=len(fnos), desc='Generating raw fnos', unit='symbol') as pbar:
        
        for s in tqdm(fnos, desc='Generating raw fnos'):
            try:
                df = equity_iv_df(n.stock_quote_fno(s))
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error for {s} - error: {e}")
                pass

            pbar.update(1)
            
    df = pd.concat(dfs, ignore_index=True)

    return df


# ---- CLEANING ---

def clean_stock_history(result: list) -> pd.DataFrame:
    """Cleans output of"""

    df = pd.concat(
        [pd.DataFrame(r.get("data")) for r in result], axis=0, ignore_index=True
    )

    # ...clean columns

    mapping = {
        "CH_SYMBOL": "nse_symbol",
        "TIMESTAMP": "date",
        "CH_OPENING_PRICE": "open",
        "CH_TRADE_HIGH_PRICE": "high",
        "CH_TRADE_LOW_PRICE": "low",
        "CH_CLOSING_PRICE": "close",
        "CH_TOT_TRADED_QTY": "qty_traded",
        "CH_TOT_TRADED_VAL": "value_traded",
        "CH_TOTAL_TRADES": "trades",
        "VWAP": "vwap",
        "updatedAt": "extracted_on",
    }

    df = df[[col for col in mapping.keys() if col in df.columns]].rename(
        columns=mapping
    )

    # ...convert column datatypes

    astype_map = {
        **{
            k: "float"
            for k in ["open", "high", "low", "close", "value_traded", "trades", "vwap"]
        },
        **{"qty_traded": "int"},
    }

    df = df.astype(astype_map)

    # ...change date columns to utc

    replace_cols = ["date", "extracted_on"]
    df1 = df[replace_cols].map(lambda x: datetime.fromisoformat(x))
    df = df.assign(date=df1.date, extracted_on=df1.extracted_on)

    return df


def clean_index_history(results: list) -> pd.DataFrame:
    """cleans index history and builds it as a dataframe"""

    df = pd.concat(
        [pd.DataFrame(json.loads(r.get("d"))) for r in results], ignore_index=True
    )

    # clean the df

    # ...drop unnecessary columns

    df = df.drop(df.columns[[0, 1]], axis=1)

    # ...rename
    df.columns = ["nse_symbol", "date", "open", "high", "low", "close"]

    # ...convert nse_symbol to IB's symbol
    df = pd.concat(
        [
            df.nse_symbol.map(IDXHISTSYMMAP).rename("symbol"),
            df,
        ],
        axis=1,
    )

    utc_dates = df.date.apply(lambda x: convert_to_utc_datetime(x, eod=True))

    df = df.assign(date=utc_dates)

    # .....convert ohlc to numeric
    convert_dict = {k: "float" for k in ["open", "high", "low", "close"]}

    df = df.astype(convert_dict)

    # .....sort by date
    df.sort_values(["nse_symbol", "date"], inplace=True, ignore_index=True)

    # .....add extract_date
    now = datetime.now()
    utc_now = now.astimezone(timezone.utc)
    df = df.assign(extracted_on=utc_now)

    return df

# --- CONVERTING ---

def nse2ib(nse_list):
    """Converts nse to ib friendly symbols"""

    subs = {"M&M": "MM", "M&MFIN": "MMFIN", "L&TFH": "LTFH", "NIFTY": "NIFTY50"}

    list_without_percent_sign = list(map(subs.get, nse_list, nse_list))

    # fix length to 9 characters
    ib_equity_fnos = [s[:9] for s in list_without_percent_sign]

    return ib_equity_fnos


def make_date_range_for_stock_history(
    symbol: str, days: int = 365, chunks: int = 50
) -> list:
    """Uses `split_dates` to make date range for stock history"""

    date_ranges = split_dates(days=days, chunks=chunks)

    series = "EQ"

    ranges = [
        {
            "symbol": symbol,
            "from": start.strftime("%d-%m-%Y"),
            "to": end.strftime("%d-%m-%Y"),
            "series": f'["{series}"]',
        }
        for start, end in date_ranges
    ]

    return ranges


def equity_iv_df(quotes: dict) -> pd.DataFrame:
    """Build a core df with symbol, undPrice, expiry, strike, volatilities, lot and price."""

    flat_data = json_normalize(quotes, sep="-")

    # get symbol, lot and underlying pricefrom quote

    symbol = quotes.get("info").get("symbol")

    try:
        lot = (quotes["stocks"][0].get("marketDeptOrderBook").get("tradeInfo").get("marketLot"))
    except IndexError as e:
        logger.error(f"No lots found for {symbol}!")

        return pd.DataFrame([])


    undPrice = quotes["underlyingValue"]

    # build the df
    df = pd.DataFrame(flat_data)

    df = pd.DataFrame(
        [
            {
                "nse_symbol": symbol,
                "ib_symbol": symbol,
                "instrument": quotes.get("stocks")[i]
                .get("metadata")
                .get("instrumentType"),
                "expiry": quotes.get("stocks")[i].get("metadata").get("expiryDate"),
                "undPrice": undPrice,
                "safe_strike": 0,
                "right": quotes.get("stocks")[i].get("metadata").get("optionType")[:1],
                "strike": quotes.get("stocks")[i].get("metadata").get("strikePrice"),
                "dte": np.nan,
                "hv": quotes.get("stocks")[i]
                .get("marketDeptOrderBook")
                .get("otherInfo")
                .get("annualisedVolatility"),
                "iv": quotes.get("stocks")[i]
                .get("marketDeptOrderBook")
                .get("otherInfo")
                .get("impliedVolatility"),
                "lot": lot,
                "price": quotes.get("stocks")[i].get("metadata").get("lastPrice"),
            }
            for i in range(len(quotes))
        ]
    )

    # Convert nse_symbol to symbol
    df = df.assign(ib_symbol=nse2ib(df.nse_symbol))

    # Convert expiry to UTC NSE eod
    df = df.assign(
        expiry=df.expiry.apply(lambda x: convert_to_utc_datetime(x, eod=True))
    )

    df = df.assign(dte=get_dte(df.expiry))

    # Convert the rest to numeric
    df = df.apply(convert_to_numeric)

    # Convert to %ge
    df.iv = df.iv / 100
    df.hv = df.hv / 100

    # Change instrument type
    instrument_dict = {
        "Stock": "STK",
        "Options": "OPT",
        "Currency": "FX",
        "Index": "IDX",
        "Futures": "FUT",
    }

    inst = df.instrument.str.split()

    s = inst.apply(lambda x: "".join(instrument_dict[item] for item in x))

    df = df.assign(instrument=s)

    df = df[df.instrument.isin(["IDXOPT", "STKOPT"])]

    return df


# --- NSE CLASSES, METHODS AND DECORATORS ---


def live_cache(app_name):
    """Caches the output for time_out specified. This is done in order to
    prevent hitting live quote requests to NSE too frequently. This wrapper
    will fetch the quote/live result first time and return the same result for
    any calls within 'time_out' seconds.

    Logic:
        key = concat of args
        try:
            cached_value = self._cache[key]
            if now - self._cache['tstamp'] < time_out
                return cached_value['value']
        except AttributeError: # _cache attribute has not been created yet
            self._cache = {}
        finally:
            val = fetch-new-value
            new_value = {'tstamp': now, 'value': val}
            self._cache[key] = new_value
            return val

    """

    def wrapper(self, *args, **kwargs):
        """Wrapper function which calls the function only after the timeout,
        otherwise returns value from the cache.

        """
        # Get key by just concating the list of args and kwargs values and hope
        # that it does not break the code :P
        inputs = [str(a) for a in args] + [str(kwargs[k]) for k in kwargs]
        key = app_name.__name__ + "-".join(inputs)
        now = datetime.now()
        time_out = self.time_out
        try:
            cache_obj = self._cache[key]
            if now - cache_obj["timestamp"] < timedelta(seconds=time_out):
                return cache_obj["value"]
        except:
            self._cache = {}
        value = app_name(self, *args, **kwargs)
        self._cache[key] = {"value": value, "timestamp": now}
        return value

    return wrapper


class NSEfnos:
    """Class for All NSE FNOS, including Indexes"""

    time_out = 5
    base_url = "https://www.nseindia.com/api"
    page_url = "https://www.nseindia.com/get-quotes/equity?symbol=LT"
    _routes = {
        "stock_meta": "/equity-meta-info",
        "stock_quote": "/quote-equity",
        "stock_derivative_quote": "/quote-derivative",
        "market_status": "/marketStatus",
        "chart_data": "/chart-databyindex",
        "market_turnover": "/market-turnover",
        "equity_derivative_turnover": "/equity-stock",
        "all_indices": "/allIndices",
        "live_index": "/equity-stockIndices",
        "index_option_chain": "/option-chain-indices",
        "equity_option_chain": "/option-chain-equities",
        "currency_option_chain": "/option-chain-currency",
        "pre_open_market": "/market-data-pre-open",
        "holiday_list": "/holiday-master?type=trading",
        "stock_history": "/historical/cm/equity",  # added by rkv
    }

    def __init__(self):
        self.s = requests.Session()
        h = {
            "Host": "www.nseindia.com",
            "Referer": "https://www.nseindia.com/get-quotes/equity?symbol=SBIN",
            "X-Requested-With": "XMLHttpRequest",
            "pragma": "no-cache",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        self.s.headers.update(h)
        self.s.get(self.page_url)

    def get(self, route, payload={}):
        url = self.base_url + self._routes[route]
        r = self.s.get(url, params=payload)
        return r.json()

    @live_cache
    def stock_quote(self, symbol):
        data = {"symbol": symbol}
        return self.get("stock_quote", data)

    @live_cache
    def stock_quote_fno(self, symbol):
        data = {"symbol": symbol}
        return self.get("stock_derivative_quote", data)

    @live_cache
    def trade_info(self, symbol):
        data = {"symbol": symbol, "section": "trade_info"}
        return self.get("stock_quote", data)

    @live_cache
    def market_status(self):
        return self.get("market_status", {})

    @live_cache
    def chart_data(self, symbol, indices=False):
        data = {"index": symbol + "EQN"}
        if indices:
            data["index"] = symbol
            data["indices"] = "true"
        return self.get("chart_data", data)

    @live_cache
    def tick_data(self, symbol, indices=False):
        return self.chart_data(symbol, indices)

    @live_cache
    def market_turnover(self):
        return self.get("market_turnover")

    @live_cache
    def eq_derivative_turnover(self, type="allcontracts"):
        data = {"index": type}
        return self.get("equity_derivative_turnover", data)

    @live_cache
    def all_indices(self):
        return self.get("all_indices")

    def live_index(self, symbol="NIFTY 50"):
        data = {"index": symbol}
        return self.get("live_index", data)

    @live_cache
    def index_option_chain(self, symbol="NIFTY"):
        data = {"symbol": symbol}
        return self.get("index_option_chain", data)

    @live_cache
    def equities_option_chain(self, symbol):
        data = {"symbol": symbol}
        return self.get("equity_option_chain", data)

    @live_cache
    def currency_option_chain(self, symbol="USDINR"):
        data = {"symbol": symbol}
        return self.get("currency_option_chain", data)

    @live_cache
    def live_fno(self):
        return self.live_index("SECURITIES IN F&O")

    @live_cache
    def pre_open_market(self, key="NIFTY"):
        data = {"key": key}
        return self.get("pre_open_market", data)

    @live_cache
    def holiday_list(self):
        return self.get("holiday_list", {})

    @live_cache
    def stock_history(self, symbol, days: int = 365, chunks: int = 50):

        date_ranges = make_date_range_for_stock_history(symbol, days, chunks)

        result = []
        for dr in date_ranges:
            result.append(self.get("stock_history", dr))

        df = clean_stock_history(result)

        return df

    def equities(self, sort_me: bool=True) -> set:

        equities_data = self.live_fno()
        equities = [kv.get("symbol") for kv in equities_data.get("data")]
        if sort_me:
            equities.sort()

        return set(equities)

    def indexes(self):
        x = "NIFTY,BANKNIFTY,MIDCPNIFTY,NIFTYNXT50,FINNIFTY"
        return set(x.split(","))


class IDXHistories:

    time_out = 5
    base_url = "https://niftyindices.com"
    idx_symbols = IDXHISTSYMMAP.values()
    url = "https://niftyindices.com/Backpage.aspx/getHistoricaldatatabletoString"

    # prepare `post` header
    post_header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.77 Safari/537.36",
        "Connection": "keep-alive",
        "sec-ch-ua": '" Not;A Brand";v="99", "Google Chrome";v="91", "Chromium";v="91"',
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "DNT": "1",
        "X-Requested-With": "XMLHttpRequest",
        "sec-ch-ua-mobile": "?0",
        "Content-Type": "application/json; charset=UTF-8",
        "Origin": "https://niftyindices.com",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
        "Referer": "https://niftyindices.com/reports/historical-data",
        "Accept-Language": "en-US,en;q=0.9,hi;q=0.8",
    }

    def __init__(self, days: int = 365) -> None:
        self.s = requests.Session()

        # update session with default headers and get the cookies
        init_header = requests.utils.default_headers()
        init_header.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.77 Safari/537.36",
            }
        )
        self.s.headers.update(init_header)
        c = self.s.get(url=self.url)
        self.cookies = c.cookies

    def get(self, payload={}):

        r = self.s.post(
            url=self.url,
            headers=self.post_header,
            cookies=self.cookies,
            data=payload,
            timeout=self.time_out,
        )

        return r.json()

    def make_histories(self, days: int = 365, chunks: int = 50):
        """Makes histories for NIFTY50 and BANKNIFTY, based on number of days provided"""

        date_ranges = split_dates(days=days, chunks=chunks)

        # idx_symbols = ["Nifty Bank", "Nifty 50"]

        # organize the payloads
        payloads = [
            {
                "cinfo": str(
                    {
                        "name": idx_symbol,
                        "startDate": s.strftime("%d-%b-%Y"),
                        "endDate": e.strftime("%d-%b-%Y"),
                        "indexName": idx_symbol,
                    }
                )
            }
            for s, e in date_ranges
            for idx_symbol in self.idx_symbols
        ]
        # get the raw jsons
        results = []

        for payload in tqdm(payloads):
            r = self.get(payload=json.dumps(payload))
            results.append(r)

        df = clean_index_history(results)

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
