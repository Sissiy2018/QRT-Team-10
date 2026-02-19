import pandas as pd
import numpy as np
import scipy.stats as stats


class PortfolioConstructor:
    def __init__(self, target_ann_vol=500000, max_adv_pct=0.025, signal_threshold=0.75, 
                 hard_volume_limit=2000000, max_gross_exposure=10000000):
        self.target_daily_vol = target_ann_vol / np.sqrt(252)
        self.max_adv_pct = max_adv_pct
        self.hard_volume_limit = hard_volume_limit
        self.signal_threshold = signal_threshold 
        self.max_gross_exposure = max_gross_exposure # NEW: Hard cap on total book size

    def generate_target_positions(self, t, signals, cov_matrix, adv_60d, betas, benchmark_ticker):
        sig_t = signals.dropna()
        if len(sig_t) < 10:
            return pd.Series(0.0, index=signals.index)
            
        # --- 1. Soft Thresholding (The Significance Metric) ---
        active_signals = np.sign(sig_t) * np.maximum(0, sig_t.abs() - self.signal_threshold)
        active_assets = active_signals[active_signals != 0].index
        if len(active_assets) < 5:
             return pd.Series(0.0, index=signals.index)
             
        # --- 2. Risk Weighting ---
        clean_cov = cov_matrix.loc[active_assets, active_assets].fillna(0.0)
        vols = np.sqrt(np.diag(clean_cov))
        vol_series = pd.Series(vols, index=active_assets).clip(lower=0.001)
        
        raw_weights = active_signals.loc[active_assets] / vol_series
        
        # --- 3. Separate and Normalize Long/Short Books ---
        longs = raw_weights[raw_weights > 0]
        shorts = raw_weights[raw_weights < 0]
        
        pos = pd.Series(0.0, index=signals.index)
        
        if len(longs) > 0 and len(shorts) > 0:
            # Force perfectly symmetric $ neutrality in abstract space
            pos.loc[longs.index] = longs / longs.sum()
            pos.loc[shorts.index] = shorts / abs(shorts.sum())
        else:
            return pos

        # --- 3. Separate and Normalize Long/Short Books ---
        longs = raw_weights[raw_weights > 0]
        shorts = raw_weights[raw_weights < 0]
        
        pos = pd.Series(0.0, index=signals.index)
        
        if len(longs) > 0 and len(shorts) > 0:
            pos.loc[longs.index] = longs / longs.sum()
            pos.loc[shorts.index] = shorts / abs(shorts.sum())
        else:
            return pos

        # --- NEW: 3.5 Abstract Beta Hedging ---
        # Calculate the hedge BEFORE scaling so we measure the vol of the truly market-neutral book
        assets_only = pos.index[pos != 0]
        abstract_beta_exposure = (pos[assets_only] * betas.loc[assets_only].fillna(1.0)).sum()
        pos[benchmark_ticker] = -abstract_beta_exposure

        # --- 4. Target Volatility Scaling & Gross Cap ---
        # Now 'pos' includes the benchmark, so 'port_vol' measures true hedged risk
        full_clean_cov = cov_matrix.loc[pos.index, pos.index].fillna(0.0)
        port_vol = np.sqrt(pos.T @ full_clean_cov @ pos)
        
        if port_vol > 0:
            scalar = self.target_daily_vol / port_vol
            
            # Gross exposure check (excluding the benchmark from the cap)
            current_abstract_gross = pos[assets_only].abs().sum() 
            max_safe_scalar = self.max_gross_exposure / current_abstract_gross if current_abstract_gross > 0 else scalar
            
            final_scalar = min(scalar, max_safe_scalar)
            pos *= final_scalar
        else:
            return pd.Series(0.0, index=signals.index)

        # --- 5. Liquidity Constraints (Strict Clipping) ---
        # We only clip the stocks, not the benchmark hedge
        max_pos = adv_60d.loc[assets_only].fillna(0.0) * self.max_adv_pct
        max_pos = max_pos.clip(upper=self.hard_volume_limit)
        
        pos.loc[assets_only] = pos.loc[assets_only].clip(lower=-max_pos, upper=max_pos)

        # --- 6. Recalculate Final Benchmark Hedge ---
        # Because we clipped the stocks, our initial hedge is slightly off. 
        # Overwrite it to perfectly match the final, clipped beta exposure.
        final_beta_exposure = (pos[assets_only] * betas.loc[assets_only].fillna(1.0)).sum()
        pos[benchmark_ticker] = -final_beta_exposure

        return pos


import pandas as pd
import numpy as np
from scipy import stats

class MLdP_PortfolioConstructor:
    """
    Marcos Lopez de Prado inspired portfolio construction.
    1. Probabilistic Bet Sizing: Maps Z-scores (statistical significance) to [-1, 1].
    2. Inverse Volatility (Risk Parity): Downweights structurally volatile stocks.
    3. Pure Volatility Scaling: Directly sizes positions in USD to hit the target annualized vol.
    4. Realistic Frictions: Capped by ADV and absolute USD limits per day.
    5. Strict Market Neutrality: Hedges realized Beta with the benchmark.
    6. Exposure Caps: Prevents mathematically optimal but practically dangerous gross exposure.
    """
    def __init__(self, target_vol_usd=500000, max_trade_usd=2000000, max_adv_pct=0.15, max_gross_usd=10000000):
        self.target_vol_usd = target_vol_usd
        self.max_trade_usd = max_trade_usd
        self.max_adv_pct = max_adv_pct
        self.max_gross_usd = max_gross_usd # NEW: Hard cap on total portfolio size

    def generate_target_positions(self, t, signals, cov_matrix, adv_t, beta_t, benchmark, current_positions):
        # --- 1. MLdP Bet Sizing (Statistical Significance) ---
        cdf_vals = pd.Series(stats.norm.cdf(signals.fillna(0)), index=signals.index)
        bet_sizes = 2 * cdf_vals - 1 

        # --- 2. Inverse Volatility Scaling ---
        variances = pd.Series(np.diag(cov_matrix), index=cov_matrix.columns)
        vols = np.sqrt(variances) + 1e-8 
        raw_weights = bet_sizes / vols
        
        # --- 3. Abstract Dollar Neutrality ---
        longs = raw_weights[raw_weights > 0]
        shorts = raw_weights[raw_weights < 0]
        
        ideal_pos = pd.Series(0.0, index=signals.index)
        
        if len(longs) > 0 and len(shorts) > 0:
            ideal_pos.loc[longs.index] = longs / longs.sum()
            ideal_pos.loc[shorts.index] = shorts / abs(shorts.sum())

        # --- 4. Direct Scaling to Target USD Volatility & EXPOSURE CAP ---
        port_var = ideal_pos.T @ cov_matrix @ ideal_pos
        port_vol = np.sqrt(port_var)
        target_daily_vol_usd = self.target_vol_usd / np.sqrt(252)
        
        dollar_scalar = target_daily_vol_usd / port_vol if port_vol > 0 else 0
        
        # FIX: Hard Gross Exposure Cap to prevent Volatility Targeting blow-ups
        current_abstract_gross = ideal_pos.abs().sum()
        if current_abstract_gross > 0:
            max_scalar = self.max_gross_usd / current_abstract_gross
            dollar_scalar = min(dollar_scalar, max_scalar) # Take the safer of the two
            
        ideal_target_dollars = ideal_pos * dollar_scalar

        # --- 5. Apply Realistic Trade Constraints ---
        curr_pos_assets = current_positions.reindex(signals.index).fillna(0.0)
        ideal_trade = ideal_target_dollars - curr_pos_assets
        
        adv_limit = adv_t.reindex(signals.index).fillna(0.0) * self.max_adv_pct
        trade_cap = np.minimum(adv_limit, self.max_trade_usd)
        
        actual_trade = ideal_trade.clip(lower=-trade_cap, upper=trade_cap)
        actual_asset_positions = curr_pos_assets + actual_trade

        # --- 6. Enforce Market Neutrality (Beta Hedging) ---
        asset_betas = beta_t.reindex(signals.index).fillna(1.0)
        realized_beta_exposure = (actual_asset_positions * asset_betas).sum()
        
        final_positions = actual_asset_positions.copy()
        final_positions[benchmark] = -realized_beta_exposure
        
        return final_positions