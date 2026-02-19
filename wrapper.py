import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt

# 1. Load the extension
%load_ext autoreload

# 2. Use mode 3 (Deep Reload). It handles classes and nested modules much better than 2.
%autoreload 3

from preprocessing import DataProcessor
# Updated imports: only grabbing what we need
from signals import ShortTermSignalGenerator, LongTermSignalGenerator, DynamicSignalBlender
from portfolio import PortfolioConstructor, MLdP_PortfolioConstructor
from backtester import Backtester

warnings.filterwarnings('ignore')


def load_sectors(file_path, tickers):
    """Load TRBC sector classification from static data CSV."""
    df = pd.read_csv(file_path)
    sectors = df.set_index('Instrument')['TRBC Economic Sector Name']
    sectors = sectors.reindex(tickers).fillna('UNKNOWN')
    return sectors

# --- 1. Pipeline Parameters ---
BENCHMARK = 'SPX'
REBALANCE_FREQ_DAYS = 2
TARGET_ANN_VOL = 500000
MAX_ADV_PCT = 0.025
TCOST_BPS = 0
DIV_TAX = 0.30

# Define file paths
path = os.path.join('.', 'Hist_data_Russel3000', 'History_price')
file_paths = [
    os.path.join(path, 'lseg_historyprice_data_20170522_to_20151208_ADVfiltered.csv'),
    os.path.join(path, 'lseg_historyprice_data_20181102_to_20170522_ADVfiltered.csv'),
    os.path.join(path, 'lseg_historyprice_data_20200420_to_20181102_ADVfiltered.csv'),
    os.path.join(path, 'lseg_historyprice_data_20210930_to_20200420_ADVfiltered.csv'),
    os.path.join(path, 'lseg_historyprice_data_20230319_to_20210930_ADVfiltered.csv'),
    os.path.join(path, 'lseg_historyprice_data_20240828_to_20230320_ADVfiltered.csv'),
    os.path.join(path, 'lseg_historyprice_data_20260214_to_20240829.csv'),
    os.path.join(os.path.join('.', 'Hist_data_Russel3000', 'Daily_new_data'), 'lseg_historyprice_data_20260218_to_20260213_ADVfiltered.csv')
]

# --- 2. Data Preprocessing ---
print("Processing Data...")
processor = DataProcessor(benchmark_ticker=BENCHMARK)
price_close, price_ret, tot_ret, div_ret, volume, volume_usd = processor.load_and_pivot(file_paths)

# Load S&P 500 benchmark as a standalone Series, then inject into panels
sp_path = os.path.join('.', 'Hist_data_Russel3000', 'S&P',
                        'lseg_historyprice_S&P500_20260215_to_20151209.csv')
sp_df = pd.read_csv(sp_path)
sp_df['Date'] = pd.to_datetime(sp_df['Date'])
sp_df = sp_df.set_index('Date').sort_index()
sp_prices = sp_df['0'].astype(float)

# Keep a standalone benchmark Series for the signal generator
benchmark_series = sp_prices.reindex(price_close.index).copy()
benchmark_series.name = BENCHMARK

# Inject benchmark into the price/return panels
price_close[BENCHMARK] = benchmark_series
price_ret[BENCHMARK] = price_close[BENCHMARK].pct_change()
tot_ret[BENCHMARK] = price_ret[BENCHMARK]
div_ret[BENCHMARK] = 0.0
volume[BENCHMARK] = 0.0
volume_usd[BENCHMARK] = 0.0

# Impute and clean
price_close_imp, price_ret_imp, tot_ret_imp = processor.impute_missing(price_close, tot_ret)
exposure_log=[]
price_ret_clean = processor.clean_outliers(price_ret_imp)
tot_ret_clean = processor.clean_outliers(tot_ret_imp)
path = os.path.join('.', 'Hist_data_Russel3000', 'History_PE')
pe_file_paths = [
    os.path.join(path, 'lseg_Price-Earning_data_20170522_to_20151208_ADVfiltered.csv'),
    os.path.join(path, 'lseg_Price-Earning_data_20181102_to_20170522_ADVfiltered.csv'),
    os.path.join(path, 'lseg_Price-Earning_data_20181102_to_20170522_ADVfiltered.csv'),
    os.path.join(path, 'lseg_Price-Earning_data_20200420_to_20181102_ADVfiltered.csv'),
    os.path.join(path, 'Lseg_Price-Earning_data_20200420_to_20210930_ADVfiltered.csv'),
    os.path.join(path, 'Lseg_Price-Earning_data_20211001_to_20230320_ADVfiltered.csv'),
    os.path.join(path, 'Lseg_Price-Earning_data_20211001_to_20230320_ADVfiltered.csv'),
    os.path.join(os.path.join('.', 'Hist_data_Russel3000', 'Daily_new_data'), 'lseg_Price-Earning_data_0260218_to_20260213_ADVfiltered.csv')
]

# Load and process
print("Loading PE Data...")
pe_raw, ey_raw = processor.load_and_pivot_pe(pe_file_paths)
# Ensure earnings_yield is perfectly aligned with the dates/tickers of our price returns
earnings_yield = ey_raw.reindex(index=price_ret.index, columns=price_ret.columns).ffill().fillna(0)

# Calculate Betas and Hedged Returns
betas, hedged_returns = processor.compute_beta_and_hedge(tot_ret_clean, price_ret_clean)

# Sanitize data before it hits the signal generator
volume = volume.fillna(0.0)
volume_usd = volume_usd.fillna(0.0)
hedged_returns = hedged_returns.fillna(0.0)
hedged_returns = hedged_returns.replace([np.inf, -np.inf], 0.0)

# --- Load Sector Data ---
static_path = os.path.join('.', 'Hist_data_Russel3000', 'Static_data',
                            'lseg_static_data_20260216.csv')
print("Loading TRBC sector data...")
sectors = load_sectors(static_path, price_close_imp.columns)
print(f"  Sectors: {sectors.nunique()} unique, {(sectors == 'UNKNOWN').sum()} unmapped")

# ... (Assume Data Loading and Preprocessing is exactly the same as before) ...

# --- 3. Signal Generation (The New Modular Approach) ---
print("\nGenerating Structural Signals...")

# Instantiate Generators
short_gen = ShortTermSignalGenerator(reversal_window=5)
# Using default windows (252 for Momentum, 252 for IC Smoothing)
long_gen = LongTermSignalGenerator() 
blender = DynamicSignalBlender()

# Generate isolated alpha streams
short_signals = short_gen.generate(hedged_returns)

# PASS IN THE REQUIRED FUNDAMENTAL DATA HERE
long_signals = long_gen.generate(hedged_returns, earnings_yield, sectors)

# Blend them based on SPX volatility
final_signals = blender.blend(short_signals, long_signals, price_ret[BENCHMARK])

print(f"  Signal coverage: {final_signals.notna().any(axis=1).sum()} / {len(final_signals)} days")

# --- 4. Iterative Portfolio Construction ---
print("\nConstructing Portfolio through time...")

adv_60d = volume_usd.rolling(window=60, min_periods=10).mean()

# Instantiate the new cleaner constructor
portfolio_constructor = PortfolioConstructor(
    target_ann_vol=TARGET_ANN_VOL,
    max_adv_pct=MAX_ADV_PCT,
    signal_threshold=0.75,     # Skips weak signals
    hard_volume_limit=2000000, # $2M absolute max per position
    max_gross_exposure=10000000 # Strict $10M total book size cap
)

all_target_positions = pd.DataFrame(0.0, index=price_ret.index, columns=price_ret.columns)
current_positions = pd.Series(0.0, index=price_ret.columns)

warmup = 252 + 60

for i, t in enumerate(price_ret.index):
    if i < warmup:
        continue

    if i % REBALANCE_FREQ_DAYS == 0:
        sig_t = final_signals.loc[t]

        if sig_t.isna().all():
            all_target_positions.loc[t] = current_positions
            continue

        cov_matrix = tot_ret_clean.loc[:t].iloc[-60:].cov()
        adv_t = adv_60d.loc[t]
        beta_t = betas.loc[t]

        # Generate new target portfolio
        current_positions = portfolio_constructor.generate_target_positions(
            t=t, 
            signals=sig_t, 
            cov_matrix=cov_matrix, 
            adv_60d=adv_t, 
            betas=beta_t, 
            benchmark_ticker=BENCHMARK
        )
        
    else:
        # Drift with daily returns
        daily_total_ret = price_ret.loc[t].fillna(0) + div_ret.loc[t].fillna(0)
        current_positions = current_positions * (1 + daily_total_ret)

    all_target_positions.loc[t] = current_positions
    
    # Exposure tracking
    assets_only = current_positions.index.difference([BENCHMARK])
    gross_asset_exposure = current_positions[assets_only].abs().sum()
    net_asset_dollar = current_positions[assets_only].sum()
    benchmark_pos = current_positions.get(BENCHMARK, 0.0)
    
    current_beta = betas.loc[t, assets_only].fillna(1.0)
    realized_beta = (current_positions[assets_only] * current_beta).sum()
    
    exposure_log.append({
        'Date': t,
        'Gross Asset Exposure': gross_asset_exposure,
        'Net Asset Dollar': net_asset_dollar,
        'Benchmark Position': benchmark_pos,
        'Net Portfolio Beta': realized_beta + benchmark_pos 
    })

# ... (End of your existing for loop) ...

# --- 5. Live Execution Export for t+1 ---
print("\n========================================")
print("         LIVE EXECUTION PREP")
print("========================================")

# Grab the very last date we iterated over
last_date = price_ret.index[-1]

print(f"Valid signals generated for date: {last_date.date()}")
print("Preparing target notionals for t+1 execution...")

# Filter out zero positions to keep the execution file clean
active_targets = current_positions[current_positions != 0].copy()

# Create the exact DataFrame format you requested
execution_df = pd.DataFrame({
    'internal_code': active_targets.index,
    'currency': 'USD',
    'target_notional': active_targets.values
})

# Optional: Round the target notionals to 2 decimal places (cents)
execution_df['target_notional'] = execution_df['target_notional'].round(2)

# Save to CSV
output_file = 'target_notionals_t_plus_1.csv'
execution_df.to_csv(output_file, index=False)

print(f"Successfully saved {len(execution_df)} target positions to {output_file}")
print("========================================\n")

# --- EXPOSURE PLOTTING ---
exposure_df = pd.DataFrame(exposure_log).set_index('Date')

fig2, ax2 = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Plot 1: Gross vs Net Dollar Exposure
exposure_df[['Gross Asset Exposure', 'Net Asset Dollar', 'Benchmark Position']].plot(ax=ax2[0], lw=2)
ax2[0].set_title('Portfolio Exposures (USD)')
ax2[0].set_ylabel('Position Size ($)')
ax2[0].axhline(0, color='black', lw=1)
ax2[0].grid(True, alpha=0.3)

# Plot 2: Net Beta Exposure (The Neutrality Check)
exposure_df['Net Portfolio Beta'].plot(ax=ax2[1], color='purple', lw=2)
ax2[1].set_title('Net Beta Exposure (Should be ~0)')
ax2[1].set_ylabel('Beta-Adjusted Dollars')
ax2[1].axhline(0, color='black', lw=1)
ax2[1].grid(True, alpha=0.3)

plt.savefig('exposure_check.png', dpi=150, bbox_inches='tight')
print("Exposure plot saved to exposure_check.png")
# --- 5. Backtesting Engine ---
print("Running Backtest Engine...")
backtester = Backtester(benchmark_ticker=BENCHMARK, tcost_bps=TCOST_BPS, div_tax_rate=DIV_TAX)
results = backtester.run(price_ret, div_ret, all_target_positions)

# --- 6. Output & Reporting ---
print("\n" + "=" * 60)
print("BACKTEST RESULTS (Rolling Linear Pipeline)")
print("=" * 60)

net_pnl = results['Net PnL']
annualized_pnl = net_pnl.mean() * 252
annualized_vol = net_pnl.std() * np.sqrt(252)
sharpe = annualized_pnl / annualized_vol if annualized_vol > 0 else 0
total_pnl = results['Cumulative PnL'].iloc[-1]

print(f"\n  Cumulative PnL:  ${total_pnl:,.2f}")
print(f"  Annualized PnL:  ${annualized_pnl:,.2f}")
print(f"  Annualized Vol:  ${annualized_vol:,.2f}")
print(f"  Sharpe Ratio:    {sharpe:.3f}")
print(f"  Total T-Costs:   ${results['T-Costs'].sum():,.2f}")
print(f"  Total Financing: ${results['Financing'].sum():,.2f}")
print(f"  Gross Price PnL: ${results['Gross Price PnL'].sum():,.2f}")
print(f"  Dividend PnL:    ${results['Dividend PnL'].sum():,.2f}")
print(f"  Hit Rate:        {((results['Net PnL'] > 0).sum() / (results['Net PnL'] != 0).sum() * 100):.2f}%")
print(f"  Correlation with Benchmark: {results['Net PnL'].corr(benchmark_series):.2f}")

# Annual breakdown
results_dated = results.copy()
results_dated.index = price_ret.index
yearly = results_dated['Net PnL'].resample('YE').sum()
print("\n  Annual PnL:")
for dt, pnl in yearly.items():
    print(f"    {dt.year}: ${pnl:,.2f}")

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

results_dated['Cumulative PnL'].plot(ax=axes[0], color='steelblue', lw=2)
axes[0].set_title('Cumulative PnL (Rolling Linear Pipeline)')
axes[0].set_ylabel('USD')
axes[0].axhline(0, color='red', ls='--', alpha=0.4)
axes[0].grid(True, alpha=0.3)

results_dated['Net PnL'].plot(ax=axes[1], color='darkgreen', alpha=0.6)
axes[1].set_title('Daily PnL')
axes[1].set_ylabel('USD')
axes[1].axhline(0, color='red', ls='--', alpha=0.4)
axes[1].grid(True, alpha=0.3)

roll_sharpe = (net_pnl.rolling(252).mean() * 252) / (net_pnl.rolling(252).std() * np.sqrt(252))
pd.Series(roll_sharpe.values, index=price_ret.index).plot(ax=axes[2], color='darkorange', lw=2)
axes[2].set_title('Rolling 1-Year Sharpe')
axes[2].set_ylabel('Sharpe')
axes[2].axhline(0, color='red', ls='--', alpha=0.4)
axes[2].axhline(1, color='green', ls='--', alpha=0.4)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('backtest_results_linear.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to backtest_results_linear.png")
plt.show()

