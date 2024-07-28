# Objectives

1. To perpare for NSE morning naked trades
2. Prepare set of utilities that could be common to NSE and SNP
3. Fully independent of IBKR, with ability to hook to IBKR when needed
4. Class based (OOP) with stock and option bots

# Left at
- run.py

# To-do
- [x] EXPROM - expected ROM variable
- [x] YAML / JSON based variables
- [x] separate out IB related functions in nakeds.ipynb
- [x] split `nakeds.py` to `utils.py`, `ibfuncs.py` and `nse.py`
- [ ] make `run.py` with `click` that assembles test nakeds
- [ ] modify an order - from df_nakeds
- [ ] cancel an order function from df_nakeds if it is ACTIVE
- [ ] mass order delete function


- [ ] Function to save IBKR Margins on undPrice closest expiry per symbol
- [ ] Option to pick up margins from offline
- [ ] Re-organize `duplicates` in xn_history and `zArchive`
<br/>

- [ ] Extend to expiries beyond earliest
- [ ] History function to generate and save
- [ ] Delta history function to generate and save
- [ ] Dataclass templates for df_opts (nakeds/targets), df_portfolio and df_orders
- [ ] Self-sufficient option bot generation

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

## Jupyterlab IDE
- `pdm run jupyter lab .`
    - if browser doesn't load jupyter close cli and run `jupyter lab build `
-  install jupyter extensions
    - `jupyterlab-code-formatter` <i> for `black` and `isort` of imports </i>
    - `jupyterlab-code-snippets` <i> for auto codes like if \__name__ == ...</i>
    - `jupyterlab-execute-time`  <i> for execution times in cells </i>
    - `jupyterlab-git` <i> for controlling git within jupyterlab </i>
    - `jupyterlab-jupytext` <i> for saving notebook to srcipts, pdfs, etc </i>
    - `jupyterlab-plotly` <i> for graphing (alternative to matplotlib) </i>