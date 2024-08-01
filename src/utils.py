# COMMON UTILITY SCRIPTS
# ======================

import glob
import math
import os
import pickle
import re
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
import pytz
import yaml
from dotenv import load_dotenv
from from_root import from_root
from ib_async import MarketOrder, Option, util
from loguru import logger
from scipy.integrate import quad
from scipy.stats import norm
from tabulate import tabulate

ROOT = from_root()


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

        print(f'\n{self.name} started at {now.strftime("%d-%b-%Y %H:%M:%S")}')

        self._start_time = datetime.now()

    def stop(self) -> None:
        if self._start_time is None:
            raise Exception(f"Timer is not running. Use .start() to start it")

        elapsed_time = datetime.now() - self._start_time

        # Extract hours, minutes, seconds from the timedelta object
        hours = elapsed_time.seconds // 3600
        minutes = (elapsed_time.seconds % 3600) // 60
        seconds = elapsed_time.seconds % 60

        print(f"\n...{self.name} took: " + f"{hours:02d}:{minutes:02d}:{seconds:02d}\n")

        self._start_time = None


class Timediff:
    """Stores time difference for file_age."""
    def __init__(self, td: timedelta, days: int, hours: int, minutes: int, seconds: float):
        self.td = td
        self.days = days
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds


# --- CONFIGURATION FROM ENVIRONMENT ----

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


# --- INTERACTIONS ---

def yes_or_no(question, default="n") -> bool:
  """Asks a yes or no question with a default answer.

  Args:
    question: The question to ask.
    default: The default answer if the user presses Enter.

  Returns:
    True if the user answered yes, False otherwise.
  """

  while True:
    answer = input(question + f" (y/n): ").lower().strip()
    if not answer:
      return default == "y"
    if answer in ("y", "yes"):
      return True
    elif answer in ("n", "no"):
      return False
    else:
      print("Please answer yes or no.")


# --- FILE HANDLING ---

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


def delete_files(file_paths):
    """Deletes a list or set of files.

    Args:
      file_paths: A list or set of file paths as strings.
    """

    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            else:
                print(f"File not found: {file_path}")
        except OSError as e:
            print(f"Error deleting file: {file_path}, {e}")


def get_pickle_suffix(pattern: str = "*nakeds*"):
    """Checks pickle file name suffix and gets the next digit"""

    def extract_suffix(file):
        try:
            return int(file[:-4][-1])
        except ValueError:
            return 0

    files = get_files_from_patterns(pattern=pattern)
    suffixes = [extract_suffix(file) for file in files]
    return max(suffixes) + 1 if suffixes else 1


def remove_raw_nakeds(save: bool=True):

    # consolidate and pickle
    files = get_files_from_patterns(pattern="*nakeds*")

    if files:
        df_nakeds = pd.concat([get_pickle(f) for f in files], ignore_index=True)

        # save a df_nakeds for rough use
        if save:
            pickle_me(df_nakeds, ROOT / "data" / "df_nakeds.pkl")

            # historize
            filename = f"{datetime.now().strftime('%Y%m%d_%I_%M_%p')}_naked_orders.pkl"
            pickle_me(df_nakeds, str(ROOT / "data" / "xn_history" / str(filename)))

        # delete consolidated files
        delete_files(files)
        logger.info(f"Deleted files {files}")


def get_files_from_patterns(log_path = None, pattern: str = "*nakeds*") -> list:
    """Gets list of files of matching pattern from a folder path """

    ROOT = from_root()

    if log_path is None: # Defaults to raw data folder
        log_path = ROOT / 'data' / 'raw'

    result = glob.glob(str(log_path / pattern))

    return result

def get_file_age(file_path: Path) -> Optional[Timediff]:
    """Gets age of a file in timedelta and d,h,m,s."""
    if not file_path.exists():
        logger.info(f"{file_path} file is not found")
        return None

    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
    time_now = datetime.now()
    td = time_now - file_time

    return split_time_difference(td)

def split_time_difference(diff: timedelta) -> Timediff:
    """Splits time difference into days, hours, minutes, seconds."""
    days = diff.days
    hours, remainder = divmod(diff.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    seconds += diff.microseconds / 1e6

    return Timediff(diff, days, hours, minutes, seconds)


def how_many_days_old(file_path) -> float:
    """Gets the file's age in days"""
    file_age = get_file_age(file_path=file_path)
    
    seconds_in_a_day = 86400
    file_age_in_days = file_age.td.total_seconds() / seconds_in_a_day if file_age else 0
    
    return file_age_in_days

def pickle_with_age_check(obj: dict, file_name_with_path: Path, minimum_age_in_days: float = 1) -> None:
    """Pickles an object after checking file age."""
    existing_file_age = get_file_age(file_name_with_path)
    
    seconds_in_a_day = 86400  # 24 * 60 * 60
    file_age_in_days = existing_file_age.td.total_seconds() / seconds_in_a_day if existing_file_age else 0

    if existing_file_age is None or file_age_in_days >= minimum_age_in_days:
        pickle_me(obj, file_name_with_path)
        logger.info(f"Pickled object to {file_name_with_path}")
    else:
        logger.info(f"Not pickled as {file_name_with_path}'s age {file_age_in_days:.2f} days is < {minimum_age_in_days}")

def handle_raws(): 
    raw_files = get_files_from_patterns()
    if raw_files:
        ans = yes_or_no("Do you want to archive raw nakeds?")
    
        if ans:
            remove_raw_nakeds(save=True)

    else:
        print("No raw files to archive")
        

# --- TRANSFORMATIONS ---

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


def chunk_me(data, size: int = 25) -> list:
    """
    Chunks the given data into smaller chunks of specified size.

    Args:
        data: The data to be chunked. Can be a list, pd.Series, pd.DataFrame, or set.
        size: The desired size of each chunk. Defaults to 25.

    Returns:
        A list of chunks, or None if the data type is not supported.
    """
    
    if isinstance(data, (list, pd.Series, pd.DataFrame)):
        return [data[i:i + size] for i in range(0, len(data), size)]
    elif isinstance(data, set):
        data_list = list(data)
        return [data_list[i:i + size] for i in range(0, len(data_list), size)]
    
    logger.error(f"Data type needs to be a list, pd.Series, pd.DataFrame, or set, not {type(data)}")
    return None



def clean_ib_util_df(contracts: Union[list, pd.Series], eod=True, ist=True) -> Union[pd.DataFrame, None]:
    """Cleans ib_async's util.df to keep only relevant columns"""
    
    if isinstance(contracts, pd.Series):
        contracts = contracts.to_list()
        
    try:
        udf = util.df(contracts)
    except (AttributeError, ValueError) as e:
        logger.error(f"cannot clean {type(contracts)} type. Have to be list or series")
        udf = None
    
    if not udf.empty:
        udf = udf[['symbol', 'conId', 'secType', 'lastTradeDateOrContractMonth','strike', 'right',]]
        
        udf.rename({"lastTradeDateOrContractMonth": "expiry"}, 
                   axis="columns", inplace=True)
        
        udf = udf.assign(expiry=udf.expiry.
                         apply(lambda x: 
                           convert_to_utc_datetime(x, eod=eod, ist=ist)))
        
        udf = udf.assign(contract=contracts)

    return udf


def convert_to_utc_datetime(date_string, eod=False, ist=True):

    """Converts string dates to UTC time.
    Args:
      date_string: in various formats
      eod: forces date to end of day for option expiries
      ist: Indian Standard Time if true. Else it is NY time"""

    formats = ["%Y%m%d", "%Y%m%d %H:%M:%S UTC", "%d-%b-%Y", "%d %b %Y", "%Y-%m-%d %H:%M:%S.%f%z", "%Y%m%d"]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_string, fmt)
            break
        except ValueError:
            pass
    else:
        raise ValueError("Invalid date string format")

    if eod:
        if ist:
            timezone = pytz.timezone("Asia/Kolkata")
            dt = dt.replace(hour=15, minute=30, second=0)
        else:
            timezone = pytz.timezone("America/New_York")
            dt = dt.replace(hour=16, minute=0, second=0)
            
        dt = timezone.localize(dt)

    utc_timezone = pytz.timezone("UTC")
    dt_utc = dt.astimezone(utc_timezone)
    
    return dt_utc


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


def clean_symbols(symbols: str) -> list:
    """Cleans a symbol symbol or a list of symbols
    Arg
      symbols: in commas eg. 'reliance, sbin' or just 'reliance' 
    
    """
    config = load_config()
    NSE2IB = config.get('NSE2IB')

    if ',' in symbols[0]:
        symbols = [s.strip().upper() for s in symbols[0].split(',')]
        
    else:
        symbols = [s.upper() for s in symbols]

    return [NSE2IB.get(s, s) for s in symbols]


def merge_and_overwrite_df(df1, df2):
  """Merges df2 into df1, overwriting common columns and preserving order.

  Args:
    df1: The base DataFrame.
    df2: The DataFrame to merge into df1.

  Returns:
    The merged DataFrame.
  """

  # Identify columns unique to df2
  new_cols = df2.columns.difference(df1.columns)

  # Merge df1 and df2 on index, using outer join to include all columns
  merged_df = pd.merge(df1, df2[new_cols], left_index=True, right_index=True, how='outer')

  # Reorder columns to match df1
  merged_df = merged_df[df1.columns.tolist() + new_cols.tolist()]

  return merged_df


# --- SEEKING ---

def get_closest_strike(df, above=None):
    """
    Finds the row with the strike closest to the undPrice.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    above (bool or None): If True, find the closest strike above undPrice.
                         If False, find the closest strike below undPrice.
                         If None, find the closest strike regardless of direction.

    Returns:
    pd.DataFrame: A DataFrame with the single row that has the closest strike.
    """
    undPrice = df["undPrice"].iloc[0]  # Get the undPrice from the first row

    if above is not None:
        # Filter for strikes above or below undPrice based on the above parameter
        mask = df["strike"] > undPrice if above else df["strike"] < undPrice
    else:
        # No filtering if above is None
        mask = None

    if mask is not None and not mask.any():
        return pd.DataFrame()  # Return an empty DataFrame if no rows match the criteria

    if mask is not None:
        df = df[mask]

    # Calculate the absolute difference between strike and undPrice
    diff = np.abs(df["strike"] - undPrice)

    # Find the index of the row with the minimum difference
    closest_index = diff.idxmin()

    # Return the closest row as a DataFrame
    return df.loc[[closest_index]]


def get_dte(s: pd.Series) -> pd.Series:
    """Gets days to expiry. Expects series of UTC timestamps"""

    now_utc = datetime.now(pytz.UTC)
    return (s - now_utc).dt.total_seconds() / (24 * 60 * 60)


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


# --- APPENDING TO DATAFRAMES ---

def append_safe_strikes(df: pd.DataFrame) -> pd.DataFrame:
    """Appends safe-strikes and intrinsics from iv, undPrice and dte"""

    config = load_config()

    PUTSTDMULT = config.get("PUTSTDMULT")
    CALLSTDMULT = config.get("CALLSTDMULT")

    sdev = pd.Series(
                df.iv
                * df.undPrice
                * (df.dte / 365).apply(lambda x: math.sqrt(x) 
                                       if x >= 0 else np.nan), name="sdev")
    
    df_sp = merge_and_overwrite_df(df, sdev.to_frame())

    # calculate safe strike with option price added
    safe_strike = np.where(
        df_sp.right == "P",
        (df_sp.undPrice - df_sp.sdev * PUTSTDMULT).astype("int"),
        (df_sp.undPrice + df_sp.sdev * CALLSTDMULT).astype("int"),
    )

    df_sp = df_sp.assign(safe_strike=safe_strike)

    # intrinsic value
    intrinsic = np.where(
        df_sp.right == "P",
        (df_sp.strike - df_sp.safe_strike).map(lambda x: max(0, x)),
        (df_sp.safe_strike - df_sp.strike).map(lambda x: max(0, x)),
    )

    df_sp = df_sp.assign(intrinsic=intrinsic)

    return df_sp


def append_black_scholes(df: pd.DataFrame, risk_free_rate: float) -> pd.DataFrame:
    """Appends black_scholed price to df
    Args:
       df: with undPrice, strike, right, iv and dte
       risk_free_rate: in decimals (e.g. 0.06 for 6%)"""

    # Compute the black_scholes of option strike
    bsPrice = df.apply(
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

    if isinstance(bsPrice, pd.DataFrame):
        if bsPrice.empty:
            df_out = df.assign(bsPrice=np.nan)
    else:
        df_out = df.assign(bsPrice=bsPrice)

    return df_out


def append_cos(df: pd.DataFrame) -> pd.DataFrame:

    """Append contract and order fields"""

    dfo = make_contracts_orders(df)
    df = df.assign(contract=dfo.contract, order=dfo.order)

    return df


def append_xPrice(df: pd.DataFrame) -> pd.DataFrame:

    """Append expected price, filter minimum rom and sort by likeliest"""

    # remove order column
    df = df.drop(columns=['order'], errors='ignore')
    
    # get maxprice
    maxPrice = np.maximum(df.price, df.bsPrice)
    
    # get expected price
    xPrice = (df.intrinsic + maxPrice).apply(lambda x: max(get_prec(x, 0.05), 0.05))
    df = df.assign(xPrice = xPrice)
    
    # prevent divide by zero for rom
    margin = np.where(df.margin <= 0, np.nan, df.margin)
    
    # calculate rom
    rom = df.xPrice * df.lot / margin * 365 / df.dte
    df = df.assign(rom=rom)

    # ensure minimum expected ROM
    config = load_config()
    MINEXPROM = config.get('MINEXPROM')
    df = df[df.rom > MINEXPROM].reset_index(drop=True)

    # sort by likeliest
    df = df.loc[(df.xPrice / df.price).sort_values().index]

    return df


# --- BUILDING ---

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


def arrange_orders(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Arranges orders to get the best ROM and filters out excessive margins
    
    Usage:
      arrange_orders(df, 
                   maxmargin=MAXMARGINPERORDER, 
                   how_many=2,
                   puts_only=True)"""

    MARKET = df.contract.iloc[0].exchange

    # Takes maxmargin if it is provided. Else defaults it to 5,000 for SNP
    maxmargin = kwargs.get('maxmargin', 5000)
    how_many = kwargs.get('how_many', 2)
    puts_only = kwargs.get('puts_only', False)

    # files = get_files_from_patterns("/*nakeds*")


    # Sort by safe_strike: strike ratio to get the safest calls and puts

    # ... group calls

    if not puts_only:
        gc = (
            df[df.right == "C"]
            .assign(ratio=df.safe_strike / df.strike)
            .sort_values("ratio")
            .groupby("ib_symbol")
        )
        df_calls = gc.head(how_many).sort_values(
            ["ib_symbol", "strike"], ascending=[True, False]
        )
        dfc = df_calls[df_calls.margin < maxmargin]
        dfc = dfc.assign(ratio=dfc.safe_strike / dfc.strike).sort_values("ratio")
    else:
        dfc = pd.DataFrame([])

    # ... group puts
    gc = (
        df[df.right == "P"]
        .assign(ratio=df.strike / df.safe_strike)
        .sort_values("ratio")
        .groupby("ib_symbol")
    )
    df_puts = gc.head(how_many).sort_values(
        ["ib_symbol", "strike"], ascending=[True, False]
    )
    dfp = df_puts[df_puts.margin < maxmargin]
    dfp = dfp.assign(ratio=dfp.strike / dfp.safe_strike).sort_values("ratio")

    # ... prepare the nakeds to order
    df_nakeds = pd.concat([dfc, dfp], axis=0, ignore_index=True)

    # ... remove dubious xPrice which are less than option price
    df_nakeds = df_nakeds[df_nakeds.xPrice > df_nakeds.price]

    df_nakeds = df_nakeds.reset_index(drop=True)

    # Sort the DataFrame by 'xPrice/price' in ascending order
    df_nakeds = df_nakeds.loc[
        df_nakeds["xPrice"].
            div(df_nakeds["price"]).
            sort_values().index
    ]

    return df_nakeds


# --- COMPUTATIONS ---

def black_scholes(
    S: float,  # underlying
    K: float,  # strike
    T: float,  # years-to-expiry
    r: float,  # risk-free rate
    sigma: float,  # implied volatility
    option_type: str,  # Put or Call right
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


# --- FORMATTING / PRETTIFYING ---

def arrange_df_columns(df: pd.DataFrame, col_map: str) -> list:
    """Extracts desired columns from df based on comma / tab / space sepeartors
    
    Args:
      df: input dataframe
      col_map: comma | space | tab separated string of columns names
    
    Returns:
      A list of ordered column names for the df
    
    """

    cols = []
    
    if isinstance(col_map, str):
        v = re.split(r"[,\t\s]+", col_map)
        cols = [s for s in v if s in df.columns]
        
    return cols


def pretty_print_df(df):
    """Pretty prints a pandas DataFrame to the console."""
    if not df.empty:
        headers = df.columns.tolist()
        print(tabulate(df, headers='keys',
            floatfmt=".2f",))
    else:
        print("Nothing to print!!\n")

def split_and_uppercase(s):
    if isinstance(s, (tuple, set)):
        # If it's a tuple or set, process each element
        result = []
        for item in s:
            result.extend(re.split(r'[,\s]+', item))
        return [item.upper() for item in result if item]
    else:
        # If it's a single string, process it directly
        return [item.upper() for item in re.split(r'[,\s]+', s) if item]

# --- TEST BENCH --- 

if __name__ == "__main__":
    ans = yes_or_no("Do you want to proceed?")
    print(ans)
