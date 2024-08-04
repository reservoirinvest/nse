# Objectives

1. Perpare for NSE morning naked trades
2. Prepare set of utilities that could be common to NSE and SNP
3. Fully independent of IBKR, with ability to hook to IBKR when needed
4. Class (OOP) based with stock and option bots

# Rules

## Symbols
-------
a. Every valid symbol should have at least one order in the system
b. Symbol without an underlying position should have one naked order
c. Underlying positions should have two options
   - i. For Put shorts: a Covered Call sell and a Protective Put buy position
   - ii. For Call Shorts:  a Covered Put sell and a Protective Call buy position    \n

d. Missing opitons should be available as orders

## Orchestrator
------------
... will be continuously running to check for the following events.
1. If the margin cushion is lower than 10% all open shorts for non-poisitions will be cancelled
2. If there is an order fill
   - selling price of all open shorts for non-positions will be bumped up
   - the order fill will be journaled
   - algo will go to recalculate mode
   - selling price of all open shorts for non-positions will be modified per re-calculation
   - algo will go to monitor (listening) mode
3. Will schedule requests for information, like history, portfolio

# Programs (sequential)
1. `naked_orders()`
   - `fnos()` ... list of fnos (weekly preferred. includes both stocks and index)
   - `bans()` ... banned stocks of the exchange
   - `underlyings()` ... get `price()`, `iv()`, `closest_opt_price()` and `closest_margin()`
   - `chains()` ... all option chains limited by a `DTEMAX` that is typically 50 days.
   - `targets()` ... `target_calls()` based on `CALLSTDMULT` and `target_puts()` based on `PUTSTDMULT` with `xPrice`

2. `cover_orders()` ... for `COVERSTD` that is typically 1 SD

3. `protect_orders()` ... for `PROTECTSTD` that is tyically 1 SD

## ---- ON DEMAND ----

4. `get_portfolio()` ... with `cushion()`, `pnl()` and `risk()`

5. `get_openorders()` 

6. `und_history()` ... OHLCs of underlyings. Updated in `delta` mode for missing days.

7. `opt_history()` ... OHLCs of options. Updated in `delta` mode for missing days.

## ---- CONTINUOUS MONITORING ----

8. EVENT: MARGIN_BREACH
9. EVENT: ORDER_FILL

10. `bump_price()` ... by 10% upon order fill
11. `recalculate()` that runs naked_orders function.


## Left at
- snp.py

## To-do

- [ ] Make `snp.py` with:
   - [ ] qualified underlyings
   - [ ] price, margins and iv for underlyings
   - [ ] chains for `df_all` options
   - [ ] 


- [ ] Option to pick up margins from offline
- [ ] Extend to expiries beyond earliest


- [ ] modify an order - from df_nakeds
- [ ] cancel an order function from df_nakeds if it is ACTIVE
- [ ] mass order delete function

<br/>

- [ ] History function to generate and save
- [ ] Delta history function to generate and save
- [ ] Dataclass templates for df_opts (nakeds/targets), df_portfolio and df_orders
- [ ] Self-sufficient continuous-monitoring and autonomous option bots

---

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

---

# Run after installation
- First activate venv with `pdm venv activate`

## Running with CLI
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