# COMMON UTILITY SCRIPTS
# ======================

import glob
import math
import os
import pickle
import re
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Union

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

        print(f'\n{self.name} started at {now.strftime("%d-%b-%Y %H: %M:%S")}')

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


def get_pickle_suffix(pattern: str = "/*nakeds*"):
    """Checks pickle file name suffix and gets the next digit"""

    def extract_suffix(file):
        try:
            return int(file[:-4][-1])
        except ValueError:
            return 0

    files = get_files_from_patterns(pattern)
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


def clean_ib_util_df(contracts: Union[list, pd.Series]) -> pd.DataFrame:
    """Cleans ib_async's util.df to keep only relevant columns"""

    df1 = pd.DataFrame([])  # initialize

    if isinstance(contracts, list):
        df1 = util.df(contracts)
    elif isinstance(contracts, pd.Series):
        try:
            contract_list = list(contracts)
            df1 = util.df(contract_list)  # it could be a series
        except (AttributeError, ValueError):
            logger.error(f"cannot clean type: {type(contracts)}")
    else:
        logger.error(f"cannot clean unknowntype: {type(contracts)}")

    if not df1.empty:

        df1.rename(
            {"lastTradeDateOrContractMonth": "expiry"}, axis="columns", inplace=True
        )

        df1 = df1.assign(expiry=pd.to_datetime(df1.expiry))
        cols = list(df1.columns[:6])
        cols.append("multiplier")
        df2 = df1[cols]
        df2 = df2.assign(contract=contracts)

    else:
        df2 = None

    return df2


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


# --- SEEKING ---

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

# --- TEST BENCH --- 

if __name__ == "__main__":
    ans = yes_or_no("Do you want to proceed?")
    print(ans)
