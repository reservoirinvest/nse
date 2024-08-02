# Objectives

1. Perpare for NSE morning naked trades
2. Prepare set of utilities that could be common to NSE and SNP
3. Fully independent of IBKR, with ability to hook to IBKR when needed
4. Class (OOP) based with stock and option bots

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