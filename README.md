# Walmart Sales Forecasting

## Installation instructions
- Clone the repo
- Download the [dataset](https://www.kaggle.com/competitions/walmart-recruiting-sales-in-stormy-weather/) and put the archive into data/raw. No need to extract it
- Install Poetry as described [here](https://python-poetry.org/docs/#installation)
- Install Python 3.9 and make sure it's active in the repo root directory by running ```python --version```. I'd recommend using [pyenv](https://github.com/pyenv/pyenv) for this
- Run ```poetry install```
- Run ```poetry run process-raw-data```
