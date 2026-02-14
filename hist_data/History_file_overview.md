## File Overview

This folder contains the code and processed data for the **Russell 3000 index** analysis.

The workflow includes:

1. Historical price data extraction
2. Daily USD volume calculation
3. Average Daily Volume (ADV) computation
4. Trading limit generation
5. Filtering eligible RICs based on liquidity criteria

------

## Data Files Description

### 1. `lseg_historyprice_data_20260214_to_20240829.csv`

This file contains **historical price data for all tickers under the Russell 3000 index** within the specified date range.

It serves as the raw input data for subsequent volume and liquidity calculations.

------

### 2. `lseg_historyprice_data_20260214_to_20240829_dailyusdvolumn`

This file adds a new column:

- `Daily_USD_Volume`

The daily USD volume is calculated using the formula:

```
Daily_USD_Volume = Volume × Closing Price
```

(See corresponding script for implementation details.)

------

### 3. `lseg_data_20260214_ADV_75days_0.8coverage`

This file contains the **Average Daily Volume (ADV)** calculation generated on the date in the filename.

- ADV is computed using the **past 75 trading days**
- Only stocks with at least **80% data coverage (0.8 coverage threshold)** are retained
- Stocks below the coverage threshold or with NAs are filtered out

The lookback window can be adjusted (e.g., to 60 days in the guideline).
However, note that **today and yesterday may not have complete data**, which may affect the calculated trading limits. See `calculate_adv` in `pull_data.ipynb`.

------

### 4. `lseg_data_20260214_trading_limit`

This file contains the **daily trading limits**, derived from the ADV calculation generated on the date in the filename.

Trading limits are computed based on:

- A percentage of ADV
- Subject to a maximum USD cap

(See  `apply_single_stock_limits` in `pull_data.ipynb` for exact formula.)

------

### 5. `rics_20260214.json`

This JSON file contains the list of **RICs that passed the ADV liquidity threshold** and are eligible for trading.

------

### 6. `ADV_filtered files`

These files are the **historical data for the selected RICs** (i.e., those meeting the ADV requirement).

Only RICs included in `rics_20260214.json` are stored here. The file for the full index will be generated but that takes longer.

------

## Summary of Workflow

1. Start from full Russell 3000 historical data
2. Compute Daily USD Volume
3. Compute ADV (75-day lookback, 80% coverage filter)
4. Apply liquidity threshold
5. Generate trading limits
6. Store eligible RICs and filtered historical datasets