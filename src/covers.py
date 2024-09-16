import numpy as np
import pandas as pd
from from_root import from_root
from ib_async import Contract, util

from ibfuncs import (get_a_price_iv, get_ib, make_chains, process_in_chunks,
                     qualify_me, quick_pf)
from snp import assemble_snp_underlyings, us_repo_rate
from utils import (append_black_scholes, get_xPrice,
                   how_many_days_old, load_config, pretty_print_df)

# Constants and Configuration
ROOT = from_root()
MARKET = 'SNP'
config = load_config(MARKET=MARKET)

# Helper Functions
def get_positions(secType: str) -> pd.DataFrame:
    with get_ib(MARKET=MARKET) as ib:
        df_pf = quick_pf(ib)

    columns = ['symbol', 'position', 'mktPrice', 'mktVal', 'avgCost', 'unPnL'] \
        if secType == 'STK' \
            else ['symbol', 'right', 'expiry', 'position', 'mktPrice', 'avgCost', 'unPnL']
    return (df_pf[df_pf['secType'] == secType][columns]
            .pipe(lambda df: df if not df.empty else pd.DataFrame(columns=columns)))

def has_covered_option(row: pd.Series, option_positions: pd.DataFrame) -> bool:
    options = option_positions[option_positions['symbol'] == row['symbol']]
    if row['position'] > 0:  # Long stock position
        return any((options['right'] == 'C') & (options['position'] < 0))  # Short call
    elif row['position'] < 0:  # Short stock position
        return any((options['right'] == 'P') & (options['position'] < 0))  # Short put
    return False

def remove_covered_positions(stock_positions: pd.DataFrame, option_positions: pd.DataFrame) -> pd.DataFrame:
    return stock_positions[
        ~stock_positions.apply(lambda row: has_covered_option(row, option_positions), axis=1)
    ]

# Main Processing Functions
def process_options(positions: pd.DataFrame, right: str) -> pd.DataFrame:
    unds_path = ROOT / 'data' / 'snp_unds.pkl'
    snp_unds = pd.read_pickle(unds_path) if how_many_days_old(unds_path) < 1 else assemble_snp_underlyings()

    filtered_unds = snp_unds[snp_unds['ib_symbol'].isin(positions['symbol'])]
    df_opts = make_chains(filtered_unds, msg='Covered chains')

    df_options = filter_options(df_opts, right)
    df_options = calculate_safe_strike(df_options, right)
    df_options = get_option_prices(df_options)

    return df_options

def filter_options(df_opts: pd.DataFrame, right: str) -> pd.DataFrame:
    max_dte = config['CCCCP_MAX_DTE']
    return df_opts[(df_opts['right'] == right) &
                   (df_opts['dte'] > 4) &
                   (df_opts['dte'] <= max_dte)].copy()

def calculate_safe_strike(df_options: pd.DataFrame, right: str) -> pd.DataFrame:
    stdmult = config['CALLSTDMULT'] if right == 'C' else config['PUTSTDMULT']
    df_options['sdev'] = df_options['und_iv'] * df_options['undPrice'] * np.sqrt(df_options['dte'] / 365)

    if right == 'C':
        df_options['safe_strike'] = np.ceil(df_options['undPrice'] + df_options['sdev'] * stdmult)
        df_options = df_options[df_options['strike'] >= df_options['safe_strike']]
        filtered = df_options.groupby('ib_symbol').apply(lambda x: x.nsmallest(2, 'strike'), include_groups=False)
    else:
        df_options['safe_strike'] = np.floor(df_options['undPrice'] - df_options['sdev'] * stdmult)
        df_options = df_options[df_options['strike'] <= df_options['safe_strike']]
        filtered = df_options.groupby('ib_symbol').apply(lambda x: x.nlargest(2, 'strike'), include_groups=False)

    return filtered.reset_index(level='ib_symbol')

def get_option_prices(df_options: pd.DataFrame) -> pd.DataFrame:
    contracts = [Contract(secType='OPT', symbol=s, lastTradeDateOrContractMonth=util.formatIBDatetime(e)[:8],
                          strike=k, right=r, exchange='SMART', currency='USD')
                 for s, e, k, r in zip(df_options.ib_symbol, df_options.expiry, df_options.strike, df_options.right)]

    with get_ib(MARKET) as ib:
        qualified_contracts = ib.run(process_in_chunks(ib, contracts, func=qualify_me,
                                                       func_args={'desc': 'qualified'}, chunk_size=200))
        prices = ib.run(process_in_chunks(ib, qualified_contracts, func=get_a_price_iv,
                                          func_args={'sleep': 15, 'gentick':''}, chunk_desc='Pricing'))

    df_price = pd.concat(prices, ignore_index=True)
    df_options = df_options.drop(['secType'], axis=1).merge(df_price, on=['ib_symbol', 'expiry', 'strike', 'right'])

    # Find the strike closest to undPrice for each ib_symbol
    df_options['strike_diff'] = abs(df_options['strike'] - df_options['undPrice'])
    df_options = df_options.loc[df_options.groupby('ib_symbol')['strike_diff'].idxmin()]
    df_options = df_options.drop('strike_diff', axis=1)

    risk_free_rate = us_repo_rate() / 100
    df_options = append_black_scholes(df_options, risk_free_rate)
    return get_xPrice(df_options)

def generate_option_recommendations(df_options: pd.DataFrame) -> pd.DataFrame:
    df_options['maxProfit'] = (abs(df_options['undPrice'] - df_options['strike']) + df_options['xPrice'])*100
    return df_options

# Main Functions
def get_covered_calls(positions: pd.DataFrame) -> pd.DataFrame:
    df_options = process_options(positions, right='C')
    return generate_option_recommendations(df_options)

def get_cash_secured_puts(positions: pd.DataFrame) -> pd.DataFrame:
    return process_options(positions, right='P')

def process_positions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stock_positions = get_positions('STK')
    option_positions = get_positions('OPT')
    stock_positions_without_covers = remove_covered_positions(stock_positions, option_positions)
    covered_calls = get_covered_calls(stock_positions_without_covers)
    return stock_positions, stock_positions_without_covers, covered_calls

if __name__ == "__main__":
    stock_positions, _, covered_calls = process_positions()

    print(f"\nRecommended Covered Calls with total maxProfit of {covered_calls.maxProfit.sum():.0f}")

    pretty_print_df(covered_calls.drop(columns=['contract', 'iv', 'hv']))
