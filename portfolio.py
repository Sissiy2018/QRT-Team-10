import pandas as pd
import numpy as np

class PortfolioConstructor:
    def __init__(self, target_ann_vol=500000, max_adv_pct=0.025):
        self.target_daily_vol = target_ann_vol / np.sqrt(252)
        self.max_adv_pct = max_adv_pct

    def generate_target_positions(self, t, signals, cov_matrix, adv_60d, betas, benchmark_ticker):
        sig_t = signals.dropna()
        if len(sig_t) < 10:
            return pd.Series(0.0, index=signals.index)
            
        top_n = max(1, int(len(sig_t) * 0.10))
        longs = sig_t.nlargest(top_n).index
        shorts = sig_t.nsmallest(top_n).index

        # --- NEW: Signal-Weighted Risk Weighting ---
        vols = np.sqrt(np.diag(cov_matrix.loc[sig_t.index, sig_t.index]))
        vol_series = pd.Series(vols, index=sig_t.index)
        
        raw_weights = pd.Series(0.0, index=sig_t.index)
        
        # Multiply inverse volatility by the absolute signal strength
        long_signal_risk = sig_t[longs].abs() / vol_series[longs]
        short_signal_risk = sig_t[shorts].abs() / vol_series[shorts]
        
        # Normalize so longs sum to 100% and shorts sum to -100% (before vol scaling)
        raw_weights[longs] = long_signal_risk / long_signal_risk.sum()
        raw_weights[shorts] = -short_signal_risk / short_signal_risk.sum()
        # ------------------------------------------

        # Initial Volatility Scaling
        port_vol = np.sqrt(raw_weights.T @ cov_matrix.loc[sig_t.index, sig_t.index] @ raw_weights)
        scalar = self.target_daily_vol / port_vol if port_vol > 0 else 0
        pos = raw_weights * scalar 

        # Iterative Liquidity Constraints & Redistribution
        max_pos = adv_60d.loc[sig_t.index] * self.max_adv_pct
        #make sure max_pos dont excceed 2000000 in size
        max_pos = max_pos.clip(upper=2000000, lower=-2000000)
        unconstrained = set(longs).union(set(shorts))
        
        for _ in range(10): # Max iterations to prevent infinite loops
            breaches = False
            for asset in list(unconstrained):
                if abs(pos[asset]) > max_pos[asset]:
                    pos[asset] = np.sign(pos[asset]) * max_pos[asset]
                    unconstrained.remove(asset)
                    breaches = True
            
            if not breaches or not unconstrained:
                break # All constraints satisfied
                
            # If there were breaches, recalculate vol and scale up unconstrained
            active = pos[pos != 0].index
            current_vol = np.sqrt(pos[active].T @ cov_matrix.loc[active, active] @ pos[active])
            
            if current_vol >= self.target_daily_vol or current_vol == 0:
                break
                
            # Scale up unconstrained assets to bridge the gap
            rescale_factor = self.target_daily_vol / current_vol
            for asset in unconstrained:
                pos[asset] *= rescale_factor

        # Calculate Benchmark Hedge Position
        assets_only = pos.index.difference([benchmark_ticker])
        beta_exposure = (pos[assets_only] * betas[assets_only]).sum()
        pos[benchmark_ticker] = -beta_exposure
        
        return pos