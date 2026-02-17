import sys
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
%load_ext autoreload
%autoreload 2
current_path= '/Users/giladfibeesh/Documents/Python/QRT-Team-10/QRT-Team-10'
sys.path.append(current_path)
# Now your imports should work!
from preprocessing import DataProcessor
from signals import Momentum12_1M
from portfolio import PortfolioConstructor
from backtester import Backtester


# --- 1. Pipeline Parameters ---
BENCHMARK = 'SPX'
REBALANCE_FREQ_DAYS = 21 # e.g., ~1 month
TARGET_ANN_VOL = 450000
MAX_ADV_PCT = 0.025
TCOST_BPS = 3
DIV_TAX = 0.30

# Define file paths (adjust to your local paths)
# Define file paths (adjust to your local paths)
# Change this line in your main() function
path = '/Users/giladfibeesh/Documents/Python/QRT-Team-10/QRT-Team-10' 

# Keep the rest exactly the same
benchmark_file = os.path.join(path,'Hist_data_Russel3000', 'S&P', 'lseg_historyprice_S&P500_20260215_to_20151209.csv')
file_paths = [
    os.path.join(path, 'Hist_data_Russel3000', 'History_price',  'lseg_historyprice_data_20170522_to_20151208_ADVfiltered.csv'),
    os.path.join(path, 'Hist_data_Russel3000', 'History_price',  'lseg_historyprice_data_20181102_to_20170522_ADVfiltered.csv'),
    os.path.join(path, 'Hist_data_Russel3000', 'History_price',  'lseg_historyprice_data_20200420_to_20181102_ADVfiltered.csv'),
    os.path.join(path, 'Hist_data_Russel3000', 'History_price',  'lseg_historyprice_data_20210930_to_20200420_ADVfiltered.csv'),
    os.path.join(path, 'Hist_data_Russel3000', 'History_price',  'lseg_historyprice_data_20230319_to_20210930_ADVfiltered.csv'),
    os.path.join(path, 'Hist_data_Russel3000', 'History_price',  'lseg_historyprice_data_20240828_to_20230320_ADVfiltered.csv'),
    os.path.join(path, 'Hist_data_Russel3000', 'History_price',  'lseg_historyprice_data_20260214_to_20240829.csv'),
]
# benchmark_file = os.path.join(path, 'Hist_data_Russel3000', 'History_price', 'lseg_historyprice_S&P500_20260215_to_20151209.csv') # <-- Add this

# --- 2. Data Preprocessing ---
print("Processing Data...")
processor = DataProcessor(benchmark_ticker=BENCHMARK)

# Pass the benchmark file as the second argument
price_close, price_ret, tot_ret, div_ret, volume_usd = processor.load_and_pivot(
    file_paths, benchmark_file
)
# Impute and clean
price_close_imp, price_ret_imp = processor.impute_missing(price_close)
tot_ret_imp = tot_ret.fillna(0) 

price_ret_clean = processor.clean_outliers(price_ret_imp)
tot_ret_clean = processor.clean_outliers(tot_ret_imp)

# Calculate Betas and Hedged Returns
betas, hedged_returns = processor.compute_beta_and_hedge(tot_ret_clean, price_ret_clean)

# --- ADD THIS LINE ---
# Prevent NaNs from slipping into the signal generator
hedged_returns = hedged_returns.fillna(0) 

# --- 3. Signal Generation ---
print("Generating Signals...")
signal_gen = Momentum12_1M()
signals = signal_gen.get_signals(hedged_returns)

# Calculate rolling 60d ADV
adv_60d = volume_usd.rolling(window=60, min_periods=10).mean()

# --- 4. Iterative Portfolio Construction ---
print("Constructing Portfolio through time...")
portfolio_constructor = PortfolioConstructor(target_ann_vol=TARGET_ANN_VOL, max_adv_pct=MAX_ADV_PCT)

# Dataframe to store the target end-of-day positions
all_target_positions = pd.DataFrame(0.0, index=price_ret.index, columns=price_ret.columns)
current_positions = pd.Series(0.0, index=price_ret.columns)

for i, t in enumerate(price_ret.index):
    # Need enough data for 12-1M momentum (252 days) and 60d covariance
    if i < 252: 
        continue
        
    if i % REBALANCE_FREQ_DAYS == 0:
            # 1. Get the last 60 days of returns
            recent_returns = tot_ret_clean.loc[:t].iloc[-60:]
            
            # 2. Find stocks that actually traded (variance > 0)
            variances = recent_returns.var()
            alive_tickers = variances[variances > 1e-8].index
            
            # 3. Filter our signals to ONLY alive stocks
            sig_t = signals.loc[t, alive_tickers].dropna()
            
            # Remove any signals that are exactly 0.0 (leftovers from fillna)
            sig_t = sig_t[sig_t.abs() > 1e-8]
            valid_tickers = sig_t.index
            
            # Optional: If your optimizer needs the benchmark in the cov matrix, keep it
            if BENCHMARK in recent_returns.columns and BENCHMARK not in valid_tickers:
                valid_tickers = valid_tickers.append(pd.Index([BENCHMARK]))

            # 4. Safety net: Do we have enough valid stocks to optimize?
            if len(valid_tickers) < 10:
                print(f"[{t.date()}] Not enough valid stocks ({len(valid_tickers)}). Going to cash.")
                current_positions = pd.Series(0.0, index=price_ret.columns)
            else:
                # 5. Create perfectly clean inputs for the optimizer
                clean_cov = recent_returns[valid_tickers].cov()
                clean_adv = adv_60d.loc[t, valid_tickers]
                clean_beta = betas.loc[t, valid_tickers]
                
                # Run the optimizer
                optimized_weights = portfolio_constructor.generate_target_positions(
                    t, sig_t, clean_cov, clean_adv, clean_beta, BENCHMARK
                )
                
                # 6. Map the optimized weights back to the massive 2938-stock universe
                current_positions = pd.Series(0.0, index=price_ret.columns)
                if optimized_weights is not None:
                    current_positions.update(optimized_weights)
    else:
        # Non-Rebalance Day: Positions drift with daily returns
        daily_total_ret = price_ret.loc[t].fillna(0) + div_ret.loc[t].fillna(0)
        current_positions = current_positions * (1 + daily_total_ret)
        
    all_target_positions.loc[t] = current_positions

# --- 5. Backtesting Engine ---
print("Running Backtest Engine...")
backtester = Backtester(benchmark_ticker=BENCHMARK, tcost_bps=TCOST_BPS, div_tax_rate=DIV_TAX)
results = backtester.run(price_ret, div_ret, all_target_positions)

# --- 6. Output & Reporting ---
print("\n--- Backtest Summary ---")

# 1. Existing Core Metrics
annualized_pnl = results['Net PnL'].mean() * 252
annualized_vol = results['Net PnL'].std() * np.sqrt(252)
sharpe = annualized_pnl / annualized_vol if annualized_vol > 0 else 0

# Filter out the warm-up period (days where PnL was exactly 0) to get accurate metrics
active_days = results[results['Net PnL'] != 0]

# 2. Hit Rate (Win Rate)
if len(active_days) > 0:
    hit_rate = (active_days['Net PnL'] > 0).sum() / len(active_days)
else:
    hit_rate = 0.0

# 3. Correlation with Benchmark
# Because correlation is scale-invariant, we can safely correlate your daily $ PnL 
# directly with the benchmark's daily % returns.
aligned_data = pd.concat([results['Net PnL'], price_ret[BENCHMARK]], axis=1).dropna()
aligned_active = aligned_data[aligned_data['Net PnL'] != 0]

if len(aligned_active) > 1:
    correlation = aligned_active['Net PnL'].corr(aligned_active[BENCHMARK])
else:
    correlation = 0.0

# 4. Maximum Drawdown
cum_pnl = results['Cumulative PnL']
running_max = cum_pnl.cummax()
drawdown = running_max - cum_pnl
max_drawdown = drawdown.max()

# Print the final tear sheet
print(f"Annualized PnL:       ${annualized_pnl:,.2f}")
print(f"Annualized Vol:       ${annualized_vol:,.2f}")
print(f"Sharpe Ratio:         {sharpe:.2f}")
print(f"Hit Rate (Win Rate):  {hit_rate * 100:.1f}%")
print(f"Max Drawdown:         ${max_drawdown:,.2f}")
print(f"Benchmark Corr:       {correlation:.2f}")
print(f"Total T-Costs:        ${results['T-Costs'].sum():,.2f}")

# Plotting: Cumulative PnL & Underwater Chart
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

# Top subplot: Cumulative PnL
ax1.plot(cum_pnl.index, cum_pnl, color='blue', label='Strategy Net PnL')
ax1.set_title('Strategy Net Cumulative PnL')
ax1.set_ylabel('USD')
ax1.grid(True)
ax1.legend()

# Bottom subplot: Drawdown (Underwater Chart)
ax2.fill_between(drawdown.index, -drawdown, 0, color='red', alpha=0.3, label='Drawdown')
ax2.set_title('Underwater Chart (Drawdowns)')
ax2.set_ylabel('USD Drop')
ax2.set_xlabel('Date')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()
