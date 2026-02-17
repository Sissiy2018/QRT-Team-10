import pandas as pd

class DataProcessor:
    def __init__(self, benchmark_ticker):
        self.benchmark_ticker = benchmark_ticker

    def load_and_pivot(self, file_paths, benchmark_file_path):
        """Loads stock CSVs and the benchmark CSV, pivoting them into unified time series."""
        
        # --- 1. Process Benchmark Data ---
        bench_df = pd.read_csv(benchmark_file_path)
        bench_df['Date'] = pd.to_datetime(bench_df['Date'])
        
        # Filter for 'Price Close' to be safe, since the column is bizarrely named 'RIC'
        bench_df = bench_df[bench_df['RIC'] == 'Price Close'].copy()
        
        # Extract the prices from column '0' and convert to a clean Series
        bench_prices = bench_df.set_index('Date')['0'].sort_index()
        bench_prices.name = self.benchmark_ticker

        # --- 2. Process Stock Data ---
        df_list = [pd.read_csv(f) for f in file_paths]
        raw_df = pd.concat(df_list, ignore_index=True)
        raw_df['Date'] = pd.to_datetime(raw_df['Date'])
        
        raw_df = raw_df.drop_duplicates(subset=['Date', 'RIC'], keep='last')
        raw_df = raw_df.sort_values(by=['Date', 'RIC'])

        price_close = raw_df.pivot(index='Date', columns='RIC', values='Price Close')
        tot_ret = raw_df.pivot(index='Date', columns='RIC', values='Daily Total Return') / 100
        volume = raw_df.pivot(index='Date', columns='RIC', values='Volume')
        
        # --- 3. Integrate Benchmark into the Universe ---
        # Outer join ensures we don't lose days where the benchmark traded but stocks didn't (or vice versa)
        price_close = price_close.join(bench_prices, how='outer')
        
        # Recalculate derived series with the benchmark included
        price_ret = price_close.pct_change()
        
        # Align indices for the other dataframes to match the new master dates
        tot_ret = tot_ret.reindex(price_close.index)
        volume = volume.reindex(price_close.index)
        
        # For the benchmark, we assume Total Return = Price Return (no dividends modeled for the index)
        tot_ret[self.benchmark_ticker] = price_ret[self.benchmark_ticker]
        
        # Calculate volume USD and Divs
        volume_usd = volume * price_close
        volume_usd[self.benchmark_ticker] = 0.0 # No volume constraint for the benchmark
        
        div_ret = tot_ret - price_ret
        div_ret[self.benchmark_ticker] = 0.0 # Ensures dividend PnL for benchmark is $0

        return price_close, price_ret, tot_ret, div_ret, volume_usd

    def impute_missing(self, price_close):
        """Forward fills prices up to a maximum of 5 days. Missing returns become 0."""
        # --- ADD LIMIT=5 HERE ---
        # If a stock doesn't trade for a week, it reverts to NaN and drops out of the optimizer
        price_close_imputed = price_close.ffill(limit=5) 
        
        price_ret_imputed = price_close_imputed.pct_change().fillna(0)
        return price_close_imputed, price_ret_imputed
    
    def clean_outliers(self, returns_df, window=60, threshold=3.5):
        """Shrinks returns > 3.5 standard deviations from 0."""
        roll_std = returns_df.rolling(window=window, min_periods=10).std()
        upper_bound = threshold * roll_std
        lower_bound = -threshold * roll_std
        
        cleaned_returns = returns_df.clip(lower=lower_bound, upper=upper_bound)
        return cleaned_returns

    def compute_beta_and_hedge(self, tot_ret_clean, price_ret_clean):
        """Computes rolling beta and returns hedged series."""
        bench_price_ret = price_ret_clean[self.benchmark_ticker]
        bench_var_250 = bench_price_ret.rolling(window=250, min_periods=50).var()
        
        betas = pd.DataFrame(index=tot_ret_clean.index, columns=tot_ret_clean.columns)
        
        for col in tot_ret_clean.columns:
            if col == self.benchmark_ticker:
                betas[col] = 1.0
                continue
                
            cov = tot_ret_clean[col].rolling(window=250, min_periods=50).cov(bench_price_ret)
            raw_beta = cov / bench_var_250
            betas[col] = 0.2 + 0.8 * raw_beta
            
        hedged_returns = tot_ret_clean.sub(betas.mul(bench_price_ret, axis=0), fill_value=0)
        return betas, hedged_returns