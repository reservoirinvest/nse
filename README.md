# Objectives

1. Perpare for morning naked trades
2. Prepare set of utilities that could be common to NSE and SNP
3. Fully independent of IBKR, with ability to hook to IBKR when needed
4. Class (OOP) based with stock and option bots

## Making now...
* tqdm for marginsAsync() with chunk_me

# To-do
- [ ] Debug `get_a_price_iv()` for block for qualified contracts in `snp_nakeds.ipynb`
- [ ] Build target options for snp.

## For SNP
- [ ] For target opts `snp_nakeds()` for `snp.py` with:
   - [x] Make `unds` with `iv` and `price`
   - [x] Make `chains` from the unds
   - [x] Get `targets` with the closest `strikes` PUTS with `undPrice` for each dte
   - [x] Make `safe_strike` of the chains with STDMULTs for the closest
   - [x] Get the option price from Black Scholes
   - [x] Rectify `process_in_chunks()` to identify errors for mass processing (like qualify_me)
   - [x] Get the market option price
   - [x] Make the xPrice with
       - * safe_strike - undPrice + strike + Black Scholes for Calls
       - * undPrice - safe_strike + strike + Black Scholes for Puts
   - [x] Get the margins and commissions for the targets
   - [x] `targets()` with appropriate standard deviation safe_strike and xPrice
   - [ ] Wrap all the above to `make_snp_nakeds()` with save
   - [ ] Make `order_snp_nakeds()`

- [ ] Option to pick up margins from offline
- [ ] Extend to expiries beyond earliest for `nse`

## General utilities
- [ ] modify an order - from df_nakeds
- [ ] cancel an order function from df_nakeds if it is ACTIVE

<br/>

- [ ] History function to generate and save
- [ ] Delta history function to generate and save
- [ ] Dataclass templates for df_opts (nakeds/targets), df_portfolio and df_orders
- [ ] Self-sufficient continuous-monitoring and autonomous option bots
- [ ] Integration to TradingView graph

---

# Rules

## Symbols
1. Every valid symbol should have at least one order in the system
2. A Symbol without an underlying position should have one naked order
3. An Underlying position should have two options:
   - For Put shorts: a Covered Call sell and a Protective Put buy position
   - For Call Shorts:  a Covered Put sell and a Protective Call buy position
4. Put and Call buys without underlying positions are `orphaned`. They should have closing orders.

## Orchestrator
... will be continuously running to check for the following events.
1. If the margin cushion is lower than 10% all open shorts for non-poisitions will be cancelled
2. If there is an order fill
   - selling price of all open shorts for non-positions will be bumped up
   - the order fill will be journaled
   - algo will go to `recalculate` mode
   - selling price of all open shorts for non-positions will be modified per re-calculation
   - algo will go to monitor (listening) mode
3. Will schedule requests for information, like `get_portfolio()` in a separate thread.

# Programs (sequential where possible)
1. `naked_orders()`
   - `fnos()` ... list of fnos (weekly preferred. includes both stocks and index)
   - `bans()` ... banned stocks of the exchange
   - `underlyings()` ... `price()`, `iv()`, `closest_opt_price()` and `closest_margin()`
   - `chains()` ... all option chains limited by a `DTEMAX` that is typically 50 days.
   - `targets()` ... `target_calls()` based on `CALLSTDMULT` and `target_puts()` based on `PUTSTDMULT` with `xPrice`
   - `place_nakeds()` ... place targets, after checking `get_portfolio()` and `get_open_orders()`

2. `opt_closures()` - create closing orders based on profitability scaled to dte from `fill_date()`. 

3. `cover_orders()` ... for stock positions with `COVERSTD` that is typically 1 SD

4. `protect_orders()` ... for stock positions with `PROTECTSTD` that is tyically 1 SD

## ---- ON DEMAND ----

1. `get_portfolio()` ... with `cushion()`, `pnl()` and `risk()`

2. `get_openorders()` 

3. `fill_date()` ... gets the order fill date from /data/xn_history (or) IB report 

4. `und_history()` ... OHLCs of underlyings. Updated in `delta` mode for missing days.

5. `opt_history()` ... OHLCs of options. Updated in `delta` mode for missing days.

## ---- CONTINUOUS MONITORING ----

1. EVENT: MARGIN_BREACH
2. EVENT: ORDER_FILL

3. `bump_price()` ... by 10% upon order fill
4. `recalculate()` ... recalculate xPrice after a re-run of `naked_orders()` function.


# Installation notes

## Folder preparation
- make a `project` folder (e.g. `nse`)
- install git with pip in it

## Virtual enviornment management
- Use `pdm` to manage virtual environment
   - use `pyproject.toml`
   - add an empty `.project-root` file at the root for relative imports / paths

### Note:
- For every package to be installed use `pdm add \<package-name> -d` 
   - the `-d` is for development environment

## Run after installation
- First activate venv with `pdm venv activate`

### Running with CLI
- Use `click` to set up. 
- See available CLI run functions with `pdm run run.py --help`
- Run the needed script with `python run.py` `<function-name>` `<--arg_name> <arg>`

## Using Jupyterlab IDE
- `pdm run jupyter lab .`
    - if browser doesn't load jupyter close cli and run `jupyter lab build `

-  install jupyter extensions
    - `jupyterlab-code-formatter` <i> for `black` and `isort` of imports </i>
    - `jupyterlab-code-snippets` <i> for auto codes like if \__name__ == ...</i>
    - `jupyterlab-execute-time`  <i> for execution times in cells </i>
    - `jupyterlab-git` <i> for controlling git within jupyterlab </i>
    - `jupyterlab-jupytext` <i> for saving notebook to srcipts, pdfs, etc </i>
    - `jupyterlab-plotly` <i> for graphing (alternative to matplotlib) </i>

- go the the directory `~/tests` and use the jupyter notebooks