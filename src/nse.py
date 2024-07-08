from datetime import date, datetime

from requests import Session

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
        "stock_history": "/api/historical/cm/equity",
        "derivatives": "/api/historical/fo/derivatives",
        "equity_quote_page": "/get-quotes/equity",
    }

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


def make_session() -> Session:

    s = Session()
    s.headers.update(h)
    s.get(page_url)

    return s