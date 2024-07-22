import json
import math
import pickle
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import requests
from bs4 import BeautifulSoup
from from_root import from_root
from ib_async import IB, MarketOrder, Option, util
from loguru import logger
from pandas import json_normalize
from scipy.integrate import quad
from scipy.stats import norm
from tqdm import tqdm

# --- CONSTANTS ---
# -----------------

ROOT = from_root()

PUTSTDMULT = 1.8
CALLSTDMULT = 2.2
PORT = port = 3000  # nse
PORT = port = 3001  # nse paper trade

# Symbol Maps
IDX_SYM_HIST_MAP = {"BANKNIFTY": "Nifty Bank", "NIFTY": "Nifty 50"}

# --- MAIN FUNCTION ---
#----------------------

def make_earliest_naked_opts(symbol: str, # nse_symbol. can be equity or index
                   ) -> pd.DataFrame:

    """Make target options for nakeds with earliest expiry"""

    # Instantiate nse
    nse = NSEfnos()
    
    # Get the basic quote
    fno_quote = nse.stock_quote_fno(symbol)

    # make the fno df with iv, hv, lot
    df_fno = equity_iv_df(fno_quote)

    # get margins and commissions from ib
    df_mcom = get_ib_margin_comms(df_fno)
    
    # Get the risk free rate
    rbi = RBI()
    risk_free_rate = rbi.repo_rate() / 100

    # Compute the expected price from black_scholes
    bsPrice = df_mcom.apply(
        lambda row: black_scholes(
            S=row["undPrice"],
            K=row["strike"],
            T=row["dte"] / 365,  # Convert days to years
            r=risk_free_rate,
            sigma=row["iv"],
            option_type=row["right"],
        ),
        axis=1,
    )

    # Adjust the expected price precision
    df = df_mcom.assign(bsPrice=bsPrice.apply(lambda x: get_prec(x, base=0.05)))
    
    # get the safe-strike, based on std multiples
    df_sp = pd.concat(
        [
            df.right,
            pd.Series(df.iv * df.undPrice * (df.dte / 365).apply(math.sqrt), name="sdev"),
            df.undPrice,
        ],
        axis=1,
    )
    
    safe_strike = np.where(
        df_sp.right == "P",
        (df_sp.undPrice - df_sp.sdev * PUTSTDMULT).astype('int'),
        (df_sp.undPrice + df_sp.sdev * CALLSTDMULT).astype('int'),
    )
    
    df = df.assign(safe_strike=safe_strike)
    
    
    # derive expected price from safe_strike
    df = df.assign(
        xPrice=abs(df.safe_strike - df.strike).apply(lambda x: get_prec(x, 0.05))
    )
    
    # calculate the return on margin (rom)
    df = df.assign(rom=df.xPrice * df.lot / df.margin * 365 / df.dte)
    
    # Sort by likeliest
    df_out = df.loc[(df.xPrice / df.price).sort_values().index]

    return df_out.reset_index(drop=True)


# --- UTILITIES ---
#----------------------


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


def split_dates(days: int = 365, chunks: int = 50) -> list:
    """splits dates into buckets, based on chunks"""

    end = datetime.today()
    periods = int(days / chunks)
    start = end - timedelta(days=days)

    if days < chunks:
        date_ranges = [(start, end)]
    else:
        dates = pd.date_range(start, end, periods).date
        date_ranges = list(
            zip(pd.Series(dates), pd.Series(dates).shift(-1) + timedelta(days=-1))
        )[:-1]

    # remove last tuple having period as NaT
    if any(pd.isna(e) for element in date_ranges for e in element):
        date_ranges = date_ranges[:-1]

    return date_ranges


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
            df.nse_symbol.map(
                {"Nifty Bank": "BANKNIFTY", "Nifty 50": "NIFTY50"}
            ).rename("symbol"),
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


def nse2ib(nse_list):
    """Converts nse to ib friendly symbols"""

    subs = {"M&M": "MM", "M&MFIN": "MMFIN", "L&TFH": "LTFH", "NIFTY": "NIFTY50"}

    list_without_percent_sign = list(map(subs.get, nse_list, nse_list))

    # fix length to 9 characters
    ib_equity_fnos = [s[:9] for s in list_without_percent_sign]

    return ib_equity_fnos


def convert_to_utc_datetime(date_string, eod=False):
    """Converts nse date strings to utc datetimes. If eod is chosen 3:30 PM IST is taken."""

    # List of possible date formats
    date_formats = ["%d-%b-%Y", "%d %b %Y", "%Y-%m-%d %H:%M:%S.%f%z"]

    for date_format in date_formats:
        try:
            dt = datetime.strptime(date_string, date_format)

            # If the parsed datetime doesn't have timezone info, assume it's UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=pytz.UTC)
            else:
                # If it has timezone info, convert to UTC
                dt = dt.astimezone(pytz.UTC)

            if eod:
                # Set time to 3:30 PM India time for all formats when eod is True
                india_time = time(hour=15, minute=30)
                india_tz = pytz.timezone("Asia/Kolkata")
                dt = india_tz.localize(datetime.combine(dt.date(), india_time))
                dt = dt.astimezone(pytz.UTC)
            elif dt.time() == time(0, 0):  # If time is midnight (00:00:00)
                # Keep it as midnight UTC
                dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)

            return dt
        except ValueError:
            continue

    # If none of the formats work, raise an error
    raise ValueError(f"Unable to parse date string: {date_string}")


def convert_to_numeric(col: pd.Series):
    """convert to numeric if possible, only for object dtypes"""

    if col.dtype == "object":
        try:
            return pd.to_numeric(col)
        except ValueError:
            return col
    return col


def convert_daily_volatility_to_yearly(daily_volatility, days: float = 252):
    return daily_volatility * math.sqrt(days)


def equity_iv_df(quotes: dict) -> pd.DataFrame:
    """Build a core df with symbol, undPrice, expiry, strike, volatilities, lot and price."""

    flat_data = json_normalize(quotes, sep="-")

    # get symbol, lot and underlying pricefrom quote

    symbol = quotes.get("info").get("symbol")

    lot = (
        quotes["stocks"][0].get("marketDeptOrderBook").get("tradeInfo").get("marketLot")
    )

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


def find_closest_strike(df, above=False):
    """
    Finds the row with the strike closest to the undPrice.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    above (bool): If True, find the closest strike above undPrice. If False, find the closest strike below undPrice.

    Returns:
    pd.DataFrame: A DataFrame with the single row that has the closest strike.
    """
    undPrice = df["undPrice"].iloc[0]  # Get the undPrice from the first row

    if above:
        # Filter for strikes above undPrice
        mask = df["strike"] > undPrice
    else:
        # Filter for strikes below undPrice
        mask = df["strike"] < undPrice

    if not mask.any():
        return pd.DataFrame()  # Return an empty DataFrame if no rows match the criteria

    # Calculate the absolute difference between strike and undPrice
    diff = np.abs(df.loc[mask, "strike"] - undPrice)

    # Find the index of the row with the minimum difference
    closest_index = diff.idxmin()

    # Return the closest row as a DataFrame
    return df.loc[[closest_index]]


def get_dte(s: pd.Series) -> pd.Series:
    """Gets days to expiry. Expects series of UTC timestamps"""

    now_utc = datetime.now(pytz.UTC)
    return (s - now_utc).dt.total_seconds() / (24 * 60 * 60)


def fbfillnas(ser: pd.Series) -> pd.Series:
    """Fills nan in series forwards first and then backwards"""

    s = ser.copy()

    # Find the first non-NaN value
    first_non_nan = s.dropna().iloc[0]

    # Fill first NaN with the first non-NaN value
    s.iloc[0] = first_non_nan

    # Fill remaining NaN values with the next valid value
    s = s.fillna(s.bfill())

    # Fill remaining NaN values with the previous valid value
    s = s.fillna(s.ffill())

    return ser.fillna(s)


def get_a_stdev(iv: float, price: float, dte: float) -> float:
    """Gives 1 Standard Deviation value for annual iv"""

    return iv * price * math.sqrt(dte / 365)


def get_prob(sd):
    """Compute probability of a normal standard deviation

    Arg:
        (sd) as standard deviation
    Returns:
        probability as a float

    """
    prob = quad(lambda x: np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi), -sd, sd)[0]
    return prob


def get_prec(v: float, base: float) -> float:
    """Gives the precision value

    Args:
       (v) as value needing precision in float
       (base) as the base value e.g. 0.05
    Returns:
        the precise value"""

    try:
        output = round(round((v) / base) * base, -int(math.floor(math.log10(base))))
    except Exception:
        output = None

    return output


# Prettify columns to show based on a dictionary map
def pretty_columns(df: pd.DataFrame, col_map: dict) -> list:
    """prettifies columns based on column map dictionary"""

    cols = [v for _, v in col_map.items() if v in df.columns]
    return cols


def get_ib_margin(contract: Option, order: MarketOrder) -> dict:
    """Gets margin and commission of a contract"""

    with IB().connect(port=port) as ib:
        if contract.conId == 0:  # qualify raw contracts
            contract = next(iter(ib.qualifyContracts(contract)))
        wif = ib.whatIfOrder(contract, order)

    # margin = float(wif.initMarginChange) # initial margin is too high compared to Zerodha, SAMCO
    margin = float(wif.maintMarginChange)
    comm = min(wif.commission, wif.minCommission, wif.maxCommission)

    return {"contract": contract, "margin": margin, "comm": comm}


def make_contracts_orders(
    df: pd.DataFrame, EXCHANGE: str = "NSE", action: str = "SELL"
) -> pd.DataFrame:
    """Makes df[contract, order] from option df. Used for margin check"""

    df_out = pd.DataFrame(
        {
            "contract": df.apply(
                lambda row: Option(
                    row.ib_symbol,
                    # util.formatIBDatetime(row.expiry.date())[:8],
                    util.formatIBDatetime(row.expiry.date()),
                    row.strike,
                    row.right,
                    EXCHANGE,
                ),
                axis=1,
            ),
            "order": df.apply(
                lambda row: MarketOrder(action=action, totalQuantity=row.lot), axis=1
            ),
        }
    )

    return df_out


def get_ib_margin_comms(df: pd.DataFrame) -> pd.DataFrame:
    
    """Qualified Contracts, Margins and Commissions from an options df"""


    symbol = df.ib_symbol.iloc[0]
    df_cos = make_contracts_orders(df)

    cts = [d if d.conId == 0 else None for d in df_cos.contract]
    with IB().connect(port=port) as ib:
        ib.qualifyContracts(*tqdm(cts, desc=f"Qualifying {symbol} options"))
        df_cos.contract = cts
        ib.disconnect()

    if len(df_cos) > 1:  # use tqdm.pandas.progress_apply()
        tqdm.pandas(desc=f"Calculating {symbol} margins")
        data = df_cos.progress_apply(
            lambda row: get_ib_margin(row.contract, row.order), axis=1
        )
    else:
        data = df_cos.apply(lambda row: get_ib_margin(row.contract, row.order), axis=1)

    df_mcom = pd.DataFrame.from_dict(data.to_dict()).T

    # replace raw contracts with qualified
    df_q = df_cos.join(df_mcom, how="outer", lsuffix="_left").drop(
        ["contract_left", "order"], axis=1
    )

    # merge margins and commissions
    df_opts = df.merge(df_q, left_index=True, right_index=True)

    # rename instrument to secType for IB
    df_opts = df_opts.assign(
        instrument=df_opts.contract.apply(lambda s: s.secType)
    ).rename(columns={"instrument": "secType"}, errors="ignore")

    return df_opts


def black_scholes(S: float, # underlying
                  K: float, # strike
                  T: float, # years-to-expiry
                  r: float, # risk-free rate
                  sigma: float, # implied volatility
                  option_type: str,# Put or Call right
                  ) -> float:
                  
    """Black-Scholes Option Pricing Model"""
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "C":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "P":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'C' for Call and 'P' for Put.")

    return price


# --- PICKLE UTILITIES ----
# -------------------------


def pickle_me(obj, file_name_with_path: Path):
    """Pickles objects in a given path"""

    with open(str(file_name_with_path), "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_pickle(path: Path, print_msg: bool = True):
    """Gets pickled object"""

    output = None  # initialize

    try:
        with open(path, "rb") as f:
            output = pickle.load(f)
    except FileNotFoundError:
        if print_msg:
            logger.error(f"file not found: {path}")

    return output


# --- CLASSES ---
# ---------------

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


    def equities(self):
        equities_data = nse.live_fno()
        equities = {kv.get("symbol") for kv in equities_data.get("data")}
        return equities

    def indexes(self):
        x = "NIFTY,BANKNIFTY,MIDCPNIFTY,NIFTYNXT50,FINNIFTY"
        return set(x.split(","))

class IDXHistories:

    time_out = 5
    base_url = "https://niftyindices.com"
    idx_symbols = IDX_SYM_HIST_MAP.values()
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


class Timer:
    """Timer providing elapsed time"""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise Exception(f"Timer is running. Use .stop() to stop it")
        
        now = datetime.now()

        print(
            f'\n{self.name} started at {now.strftime("%d-%b-%Y %H: %M:%S")}'
        )

        self._start_time = datetime.now()

    def stop(self) -> None:
        if self._start_time is None:
            raise Exception(f"Timer is not running. Use .start() to start it")

        elapsed_time = datetime.now() - self._start_time

        # Extract hours, minutes, seconds from the timedelta object
        hours = elapsed_time.seconds // 3600
        minutes = (elapsed_time.seconds % 3600) // 60
        seconds = elapsed_time.seconds % 60        

        print(
            f"\n...{self.name} took: "
            + f"{hours:02d}:{minutes:02d}:{seconds:02d}\n"
        )

        self._start_time = None



if __name__ == "__main__":

    timer = Timer("Earliest nakeds")
    timer.start()
    
    # Initialize FnO class
    nse = NSEfnos()

    fnos = nse.indexes() | nse.equities()

    dfs = []

    for symbol in fnos:

        try:
            df_nakeds = make_earliest_naked_opts(symbol)
        except AttributeError as e:
            logger.error(e)
            df_nakeds = None
            
        dfs.append(df_nakeds)

        df = pd.concat(dfs, axis=0, ignore_index=True)

        pickle_me(df, ROOT / 'data' / 'earliest_nakeds.pkl')

    timer.stop()
