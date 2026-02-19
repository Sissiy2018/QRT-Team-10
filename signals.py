from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from scipy.stats import norm

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor


class SignalGenerator:
    def get_signals(self, **kwargs):
        raise NotImplementedError("Must implement get_signals")


class Momentum12_1M(SignalGenerator):
    def get_signals(self, hedged_returns):
        log_returns = np.log1p(hedged_returns)

        # Add min_periods! E.g., require at least 200 valid days out of 252
        mom_12m = log_returns.rolling(window=252, min_periods=200).sum()
        mom_1m = log_returns.rolling(window=21, min_periods=15).sum()

        signal = mom_12m - mom_1m
        return signal


# =============================================================================
# Config for the complex ML signal pipeline
# =============================================================================

@dataclass(frozen=True)
class SignalConfig:
    # Universe / liquidity filters
    adv_window: int = 60
    min_adv_dollars: float = 5e6
    min_price: float = 2.0
    min_coverage: float = 0.70

    # Trading limits
    adv_frac_limit: float = 0.025
    max_pos_usd: float = 2e6

    # Risk management
    risk_window: int = 60
    risk_limit_usd: float = 500_000.0
    risk_budget_usd: float = 400_000.0

    # Beta estimation
    beta_window: int = 250
    beta_shrink_a: float = 0.2
    beta_shrink_b: float = 0.8

    # Walk-forward model training
    train_window: int = 756
    val_window: int = 126
    refit_every: int = 21
    min_names: int = 100

    # Portfolio construction
    name_cap_weight: float = 0.02
    eta: float = 0.08
    max_turnover: float = 0.15

    # Signal / alpha parameters
    mom_horizons: Tuple[int, ...] = (21, 63, 126, 252)
    mom_weights: Tuple[float, ...] = (0.10, 0.25, 0.35, 0.30)
    mom_skip: int = 5
    mom_vol_window: int = 63
    mom_ema_alpha: float = 0.10

    strev_window: int = 5
    lowvol_window: int = 60
    high52_window: int = 252
    amihud_window: int = 60

    # Target variable
    fwd_return_days: int = 2

    # Robust cross-sectional transforms
    winsor_k: float = 5.0
    gauss_cap: float = 3.0


# =============================================================================
# Utility functions for the complex pipeline
# =============================================================================

def _log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))


def _cs_mad_winsorize(x: pd.DataFrame, k: float) -> pd.DataFrame:
    med = x.median(axis=1)
    mad = (x.sub(med, axis=0)).abs().median(axis=1)
    mad = mad.replace(0.0, np.nan)
    lo = med - k * 1.4826 * mad
    hi = med + k * 1.4826 * mad
    return x.clip(lower=lo, upper=hi, axis=0)


def _cs_rank_gauss(x: pd.DataFrame, cap: float, eps: float = 1e-6) -> pd.DataFrame:
    u = x.rank(axis=1, method="average", pct=True).clip(eps, 1 - eps)
    z = pd.DataFrame(norm.ppf(u), index=x.index, columns=x.columns)
    return z.clip(-cap, cap)


def _cs_robust(x: pd.DataFrame, winsor_k: float, gauss_cap: float) -> pd.DataFrame:
    return _cs_rank_gauss(_cs_mad_winsorize(x, winsor_k), gauss_cap)


def _safe_div(a: pd.DataFrame, b: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    return a / (b.replace(0.0, np.nan) + eps)


def _adv_dollars(prices: pd.DataFrame, volume: pd.DataFrame, window: int) -> pd.DataFrame:
    dv = (prices * volume).replace([np.inf, -np.inf], np.nan)
    return dv.rolling(window, min_periods=window).mean()


def _build_universe_mask(prices: pd.DataFrame, volume: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    px_ok = prices >= cfg.min_price
    adv = _adv_dollars(prices, volume, cfg.adv_window)
    adv_ok = adv >= cfg.min_adv_dollars
    coverage = prices.notna().rolling(cfg.adv_window, min_periods=cfg.adv_window).mean()
    cov_ok = coverage >= cfg.min_coverage
    return (px_ok & adv_ok & cov_ok)


def _position_caps_usd(prices: pd.DataFrame, volume: pd.DataFrame, cfg: SignalConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    adv = _adv_dollars(prices, volume, cfg.adv_window)
    trade_cap = cfg.adv_frac_limit * adv
    pos_cap = np.minimum(cfg.max_pos_usd, trade_cap)
    return pos_cap, trade_cap


def _rolling_beta(prices: pd.DataFrame, benchmark: pd.Series, cfg: SignalConfig) -> pd.DataFrame:
    r = _log_returns(prices)
    rb = _log_returns(benchmark.to_frame("b"))["b"].reindex(r.index)
    var_b = rb.rolling(cfg.beta_window, min_periods=cfg.beta_window).var(ddof=0)
    cov = r.rolling(cfg.beta_window, min_periods=cfg.beta_window).cov(rb)
    raw = cov.div(var_b, axis=0)
    return cfg.beta_shrink_a + cfg.beta_shrink_b * raw


def _sector_dummies(tickers: pd.Index, sectors: pd.Series) -> pd.DataFrame:
    s = sectors.reindex(tickers).fillna("UNKNOWN")
    return pd.get_dummies(s).reindex(index=tickers).fillna(0.0).astype(float)


def _cs_neutralize_panel(
    x: pd.DataFrame,
    sectors: Optional[pd.Series],
    beta: Optional[pd.DataFrame],
    mask: Optional[pd.DataFrame],
    ridge: float = 1e-6,
    min_names: int = 120,
) -> pd.DataFrame:
    x_val = x.values
    mask_val = mask.values if mask is not None else np.ones_like(x_val, dtype=bool)
    beta_val = beta.values if beta is not None else None
    
    # --- THE FIX: Conditional Sector Dummies ---
    use_sectors = sectors is not None
    if use_sectors:
        Dfull = _sector_dummies(x.columns, sectors).values
    
    out = np.full_like(x_val, np.nan, dtype=float)
    
    for i in range(x.shape[0]):
        yv = x_val[i]
        
        m = mask_val[i] & ~np.isnan(yv) & np.isfinite(yv)
        if beta_val is not None:
            m = m & ~np.isnan(beta_val[i]) & np.isfinite(beta_val[i])
        
        if m.sum() < min_names:
            continue

        y_sub = yv[m]
        X_parts = [np.ones((m.sum(), 1))]
        
        if use_sectors:
            X_parts.append(Dfull[m])
            
        if beta_val is not None:
            X_parts.append(beta_val[i, m].reshape(-1, 1))

        X = np.concatenate(X_parts, axis=1).astype(float)

        XtX = X.T @ X
        XtX.flat[:: XtX.shape[0] + 1] += ridge
        coef = np.linalg.solve(XtX, X.T @ y_sub)
        out[i, m] = y_sub - X @ coef

    return pd.DataFrame(out, index=x.index, columns=x.columns)


def _cs_neutralize_panel(
    x: pd.DataFrame,
    sectors: Optional[pd.Series],
    beta: Optional[pd.DataFrame],
    mask: Optional[pd.DataFrame],
    ridge: float = 1e-6,
    min_names: int = 120,
) -> pd.DataFrame:
    x_val = x.values
    mask_val = mask.values if mask is not None else np.ones_like(x_val, dtype=bool)
    beta_val = beta.values if beta is not None else None
    
    # --- THE FIX: Conditional Sector Dummies ---
    use_sectors = sectors is not None
    if use_sectors:
        Dfull = _sector_dummies(x.columns, sectors).values
    
    out = np.full_like(x_val, np.nan, dtype=float)
    
    for i in range(x.shape[0]):
        yv = x_val[i]
        
        m = mask_val[i] & ~np.isnan(yv) & np.isfinite(yv)
        if beta_val is not None:
            m = m & ~np.isnan(beta_val[i]) & np.isfinite(beta_val[i])
        
        if m.sum() < min_names:
            continue

        y_sub = yv[m]
        X_parts = [np.ones((m.sum(), 1))]
        
        if use_sectors:
            X_parts.append(Dfull[m])
            
        if beta_val is not None:
            X_parts.append(beta_val[i, m].reshape(-1, 1))

        X = np.concatenate(X_parts, axis=1).astype(float)

        XtX = X.T @ X
        XtX.flat[:: XtX.shape[0] + 1] += ridge
        coef = np.linalg.solve(XtX, X.T @ y_sub)
        out[i, m] = y_sub - X @ coef

    return pd.DataFrame(out, index=x.index, columns=x.columns)


def _residualize_daily_returns(
    prices: pd.DataFrame,
    benchmark: pd.Series,
    sectors: Optional[pd.Series],
    mask: Optional[pd.DataFrame],
    min_names: int,
) -> pd.DataFrame:
    r = _log_returns(prices)
    rb = _log_returns(benchmark.to_frame("b"))["b"].reindex(r.index)
    
    r_val = r.values
    rb_val = rb.values
    mask_val = mask.values if mask is not None else np.ones_like(r_val, dtype=bool)
    
    # --- THE FIX: Conditional Sector Dummies ---
    use_sectors = sectors is not None
    if use_sectors:
        Dfull = _sector_dummies(r.columns, sectors).values
    
    out = np.full_like(r_val, np.nan, dtype=float)
    
    for i in range(r.shape[0]):
        yv = r_val[i]
        rb_i = rb_val[i]
        
        if np.isnan(rb_i):
            continue
            
        m = mask_val[i] & ~np.isnan(yv) & np.isfinite(yv)
        if m.sum() < min_names:
            continue

        y_sub = yv[m]
        X_parts = [np.ones((m.sum(), 1)), np.full((m.sum(), 1), float(rb_i))]
        
        if use_sectors:
            X_parts.append(Dfull[m])
            
        X = np.concatenate(X_parts, axis=1)
        
        coef, *_ = np.linalg.lstsq(X, y_sub, rcond=None)
        out[i, m] = y_sub - X @ coef
        
    return pd.DataFrame(out, index=r.index, columns=r.columns)


def _predict_panel(
    model, X: Dict[str, pd.DataFrame], dates: pd.DatetimeIndex,
    uni: pd.DataFrame, cfg: SignalConfig,
) -> pd.DataFrame:
    feat_names = list(X.keys())
    first_X = next(iter(X.values()))
    tickers = first_X.columns
    
    out = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    
    date_locs = first_X.index.get_indexer(dates)
    valid_mask = date_locs >= 0
    valid_dates = dates[valid_mask]
    valid_locs = date_locs[valid_mask]
    
    X_vals = [X[f].values for f in feat_names]
    uni_val = uni.fillna(False).values
    
    for t, i in zip(valid_dates, valid_locs):
        Xd = np.column_stack([arr[i] for arr in X_vals])
        ok = np.isfinite(Xd).all(axis=1) & uni_val[i]
        if ok.sum() < cfg.min_names:
            continue
        out.loc[t, tickers[ok]] = model.predict(Xd[ok])
        
    return out

# =============================================================================
# Individual alpha signal functions
# =============================================================================

def _sig_momentum(resid_r: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    vol = resid_r.rolling(cfg.mom_vol_window, min_periods=cfg.mom_vol_window).std(ddof=0)
    comps = []
    for H, wH in zip(cfg.mom_horizons, cfg.mom_weights):
        mom = resid_r.shift(cfg.mom_skip).rolling(H, min_periods=H).sum()
        comps.append(wH * _cs_robust(_safe_div(mom, vol), cfg.winsor_k, cfg.gauss_cap))
    s = sum(comps).ewm(alpha=cfg.mom_ema_alpha, adjust=False).mean()
    return s


def _sig_st_reversal(resid_r: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    # Use getattr to safely fallback just in case the config is missing the attribute
    win = getattr(cfg, 'strev_window', 5)
    # Relax min_periods to half the window so a single holiday doesn't nuke the signal
    x = -(resid_r.rolling(win, min_periods=max(1, int(win / 2))).sum())
    return _cs_robust(x, cfg.winsor_k, cfg.gauss_cap)

def _sig_lowvol(prices: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    win = getattr(cfg, 'lowvol_window', 21)
    # Inlining the log return formula here just to guarantee no downstream indexing bugs
    r = np.log(prices / prices.shift(1))
    v = r.rolling(win, min_periods=max(1, int(win / 2))).std(ddof=0)
    return _cs_robust(-v, cfg.winsor_k, cfg.gauss_cap)


def _sig_52w_high(prices: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    hi = prices.rolling(cfg.high52_window, min_periods=cfg.high52_window).max()
    x = prices / hi - 1.0
    return _cs_robust(x, cfg.winsor_k, cfg.gauss_cap)


def _sig_amihud(prices: pd.DataFrame, volume: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    win = getattr(cfg, 'amihud_window', 21)
    r_abs = np.abs(np.log(prices / prices.shift(1)))
    dv = (prices * volume).replace(0.0, np.nan)
    a = (r_abs / dv).rolling(win, min_periods=max(1, int(win / 2))).mean()
    return _cs_robust(a, cfg.winsor_k, cfg.gauss_cap)


def _sig_volume_momentum(volume: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    # Volume is highly susceptible to NaN contagion. Lower min_periods.
    vol_short = volume.rolling(21, min_periods=10).mean()
    vol_long = volume.rolling(63, min_periods=30).mean()
    x = _safe_div(vol_short, vol_long) - 1.0
    return _cs_robust(x, cfg.winsor_k, cfg.gauss_cap)

def _sig_return_consistency(resid_r: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    pos_days = (resid_r > 0).astype(float)
    # CRITICAL: Put the NaNs back so we don't calculate win-rates for dead stocks!
    pos_days[resid_r.isna()] = np.nan 
    
    win_rate = pos_days.rolling(63, min_periods=30).mean()
    return _cs_robust(win_rate - 0.5, cfg.winsor_k, cfg.gauss_cap)

def _build_features(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    benchmark: pd.Series,
    sectors: pd.Series,
    fundamentals_panels: Optional[Dict[str, pd.DataFrame]],
    uni: pd.DataFrame,
    beta: pd.DataFrame,
    cfg: SignalConfig,
) -> Dict[str, pd.DataFrame]:
    
    print("  -> Residualizing returns (this takes a minute)...")
    resid_r = _residualize_daily_returns(prices, benchmark, sectors, mask=uni, min_names=cfg.min_names)

    feats: Dict[str, pd.DataFrame] = {}
    print("  -> Building base momentum and volume signals...")
    feats["mom"] = _sig_momentum(resid_r, cfg)
    feats["strev"] = _sig_st_reversal(resid_r, cfg)
    feats["lowvol"] = _sig_lowvol(prices, cfg)
    feats["high52"] = _sig_52w_high(prices, cfg)
    feats["amihud"] = _sig_amihud(prices, volume, cfg)
    feats["vol_mom"] = _sig_volume_momentum(volume, cfg)
    feats["win_rate"] = _sig_return_consistency(resid_r, cfg)

    # ONLY neutralize features built on raw prices/volume!
    # Features built on resid_r are already neutral. Double-neutralizing 
    # Rank-Gaussed data destroys the signal variance.
    print("  -> Sector & Beta neutralizing raw features...")
    raw_feats = ["lowvol", "high52", "amihud"] 
    
    for k in tqdm(raw_feats, desc="Neutralizing Raw Features"):
        feats[k] = feats[k].reindex(index=prices.index, columns=prices.columns)
        feats[k] = _cs_neutralize_panel(feats[k], sectors=sectors, beta=beta, mask=uni, min_names=cfg.min_names)

    print("\n--- FEATURE COVERAGE X-RAY ---")
    print(f"Universe mask (avg stocks per day): {uni.sum(axis=1).mean():.1f}")
    for k, df in feats.items():
        valid_points = df.notna().sum().sum()
        print(f"  {k}: {valid_points} valid data points")
    print("------------------------------\n")
    return feats


def _build_target(
    prices: pd.DataFrame,
    benchmark: pd.Series,
    sectors: pd.Series,
    uni: pd.DataFrame,
    cfg: SignalConfig,
) -> pd.DataFrame:
    resid_r = _residualize_daily_returns(prices, benchmark, sectors, mask=uni, min_names=cfg.min_names)
    fwd = resid_r.rolling(cfg.fwd_return_days).sum().shift(-cfg.fwd_return_days)
    
    # --- THE FIX: Align target with Rank IC metric ---
    # We rank-Gauss the forward returns so outliers don't blow up the MSE loss.
    # A 5% move in a quiet market and a 15% move in a volatile market 
    # are both simply treated as "top decile" targets.
    return _cs_robust(fwd, cfg.winsor_k, cfg.gauss_cap)

# =============================================================================
# ML model and ensemble helpers
# =============================================================================

def _build_models(random_state: int = 0) -> Dict[str, Pipeline]:
    # --- THE FIX: Fast 0.0 Imputation ---
    # Rank-Gaussed features naturally have a median of 0.0. 
    imp = SimpleImputer(strategy="constant", fill_value=0.0)
    
    return {
        "ridge": Pipeline([
            ("imp", imp),
            ("scaler", StandardScaler()),
            ("m", Ridge(alpha=10.0)),
        ]),
        "ridge_heavy": Pipeline([
            ("imp", imp),
            ("scaler", StandardScaler()),
            ("m", Ridge(alpha=100.0)),
        ]),
        "hgb": Pipeline([
            ("imp", imp),
            ("m", HistGradientBoostingRegressor(
                max_depth=3,
                learning_rate=0.03,
                max_leaf_nodes=15,
                min_samples_leaf=200,
                random_state=random_state,
            )),
        ]),
    }


def _panel_to_samples(
    X: Dict[str, pd.DataFrame],
    y: pd.DataFrame,
    dates: pd.DatetimeIndex,
    uni: pd.DataFrame,
    cfg: SignalConfig,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, list]:
    feat_names = list(X.keys())
    tickers = y.columns
    rows, targs, meta = [], [], []

    for t in dates:
        if t not in y.index:
            continue
        Xd = np.column_stack([X[f].loc[t].reindex(tickers).values for f in feat_names])
        yd = y.loc[t].reindex(tickers).values
        ok = np.isfinite(Xd).all(axis=1) & np.isfinite(yd)
        ok &= uni.loc[t].reindex(tickers).fillna(False).values
        if ok.sum() < cfg.min_names:
            continue
        rows.append(Xd[ok])
        targs.append(yd[ok])
        meta.extend([(t, tickers[i]) for i in np.where(ok)[0]])

    if not rows:
        # Return explicitly 2D empty array for X, 1D for y
        return np.empty((0, len(feat_names))), np.empty((0,)), pd.DataFrame(columns=["date", "ticker"]), feat_names

    return np.vstack(rows), np.concatenate(targs), pd.DataFrame(meta, columns=["date", "ticker"]), feat_names

def _predict_panel(
    model, X: Dict[str, pd.DataFrame], dates: pd.DatetimeIndex,
    uni: pd.DataFrame, cfg: SignalConfig,
) -> pd.DataFrame:
    feat_names = list(X.keys())
    tickers = next(iter(X.values())).columns
    out = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    for t in dates:
        Xd = np.column_stack([X[f].loc[t].reindex(tickers).values for f in feat_names])
        ok = np.isfinite(Xd).all(axis=1) & uni.loc[t].reindex(tickers).fillna(False).values
        if ok.sum() < cfg.min_names:
            continue
        out.loc[t, tickers[ok]] = model.predict(Xd[ok])
    return out


def _daily_spearman_ic(pred: pd.DataFrame, y: pd.DataFrame, min_names: int) -> pd.Series:
    idx = pred.index.intersection(y.index)
    out = []
    for t in idx:
        a, b = pred.loc[t], y.loc[t]
        m = a.notna() & b.notna() & np.isfinite(a.values) & np.isfinite(b.values)
        if m.sum() < min_names:
            out.append(np.nan)
            continue
        out.append(a[m].rank().corr(b[m].rank()))
    return pd.Series(out, index=idx)


def _ic_stats(ic: pd.Series) -> Tuple[float, float, float]:
    mu = ic.mean(skipna=True)
    sd = ic.std(skipna=True, ddof=0)
    ir = mu / sd if (sd and np.isfinite(sd) and sd > 0) else np.nan
    return float(mu), float(sd), float(ir)


def _ensemble_weights_from_val(d: Dict[str, dict]) -> Dict[str, float]:
    raw = {}
    for name, info in d.items():
        mu, sd, ir = info["mean_ic"], info["std_ic"], info["ir"]
        score = ir if np.isfinite(ir) else mu
        raw[name] = max(0.0, float(score)) if np.isfinite(score) else 0.0
        
    s = sum(raw.values())
    if s <= 0:
        # --- THE FIX: Stop funding losers ---
        # If all models are negative in validation, output 0 weights.
        # This acts as an automatic risk-off switch during regime changes.
        return {k: 0.0 for k in raw}
        
    return {k: v / s for k, v in raw.items()}


def _combine_predictions(preds: Dict[str, pd.DataFrame], w: Dict[str, float]) -> pd.DataFrame:
    out = None
    for k, pk in preds.items():
        wk = w.get(k, 0.0)
        if wk == 0:
            continue
        out = pk * wk if out is None else out.add(pk * wk, fill_value=np.nan)
        
    # --- THE FIX ---
    # If all weights were 0.0 (regime shift detected), out is still None.
    # We must return a DataFrame of pure 0.0s matching the shape of the predictions.
    if out is None:
        first_pred = next(iter(preds.values()))
        out = pd.DataFrame(0.0, index=first_pred.index, columns=first_pred.columns)
        
    return out


# =============================================================================
# ComplexMLSignal: walk-forward ML ensemble signal generator
# =============================================================================

class ComplexMLSignal(SignalGenerator):
    """Walk-forward ML ensemble signal that produces alpha scores.

    Builds 7 technical features (momentum, reversal, low-vol, 52w-high,
    Amihud, volume momentum, win-rate), trains a Ridge + HGB ensemble
    via walk-forward validation, and returns sector/beta-neutralised
    alpha predictions as a DataFrame (dates x tickers).

    Usage in wrapper.py:
        signal_gen = ComplexMLSignal()
        signals = signal_gen.get_signals(
            hedged_returns,
            prices=price_close_imp,
            volume=volume_usd,
            benchmark=benchmark_series,
            sectors=sectors_series,
        )
    """

    def __init__(self, cfg: Optional[SignalConfig] = None, random_state: int = 42):
        self.cfg = cfg or SignalConfig()
        self.random_state = random_state
        self.diagnostics_ = {}

    def get_signals(
        self,
        hedged_returns,
        prices=None,
        volume=None,
        benchmark=None,
        sectors=None,
        fundamentals_panels=None,
    ):
        cfg = self.cfg

        prices = prices.sort_index()
        volume = volume.reindex_like(prices).sort_index()
        benchmark = benchmark.reindex(prices.index).sort_index()
        # sectors = sectors.reindex(prices.columns)
        sectors =None
        # Step 1: Universe + caps
        uni = _build_universe_mask(prices, volume, cfg)

        # Step 2: Rolling beta
        beta = _rolling_beta(prices, benchmark, cfg)

        # Step 3: Features and target
        X = _build_features(prices, volume, benchmark, sectors, fundamentals_panels, uni, beta, cfg)
        y = _build_target(prices, benchmark, sectors, uni, cfg)

        models = _build_models(random_state=self.random_state)

        dates = prices.index
        tickers = prices.columns
        pred_all = pd.DataFrame(index=dates, columns=tickers, dtype=float)
        ensemble_hist = {}
        ic_hist = {}

        # Step 4: Walk-forward schedule
        start = max(cfg.train_window, cfg.adv_window, cfg.beta_window,
                     cfg.mom_vol_window, cfg.high52_window) + 5
        refits = list(range(start, len(dates), cfg.refit_every))

        for rp in tqdm(refits, desc="Walk-Forward ML Refits"):
            # We are standing at day 'rp', predicting the block [rp : rp + cfg.refit_every]
            
            # The absolute latest day a 5-day forward return could be fully known
            latest_known_y_idx = rp - cfg.fwd_return_days - 1
            
            # If we don't have enough history yet to form train/val sets, skip.
            if latest_known_y_idx < (cfg.train_window + cfg.val_window + cfg.fwd_return_days):
                continue
                
            va1_idx = latest_known_y_idx
            va0_idx = va1_idx - cfg.val_window
            
            # We need another fwd_return_days buffer between train and val
            # so the end of training targets don't overlap with the start of validation prices
            tr1_idx = va0_idx - cfg.fwd_return_days
            tr0_idx = tr1_idx - cfg.train_window
            
            tr0 = dates[max(0, tr0_idx)]
            tr1 = dates[max(0, tr1_idx)]
            va0 = dates[max(0, va0_idx)]
            va1 = dates[max(0, va1_idx)]

            # 1. Training Set (Strictly past data)
            train_dates = dates[(dates >= tr0) & (dates < tr1)]
            
            # 2. Validation Set (Strictly past data, post-training)
            val_dates = dates[(dates >= va0) & (dates <= va1)]
            
            # 3. Fit Set (Train + Val combined to train the final model for 'Today')
            fit_dates = dates[(dates >= tr0) & (dates <= va1)]

            if len(train_dates) < 50 or len(val_dates) < 20:
                continue
            
            # ... (Keep the rest of your loop exactly as it is!) ...

            Xtr, ytr, _, _ = _panel_to_samples(X, y, train_dates, uni, cfg)

            if Xtr.shape[0] == 0:
                # print(f"  [!] Skipping validation at {t_refit.date()}: No data.")
                continue

            diag = {}
            for name, mdl in models.items():
                mdl.fit(Xtr, ytr)
                pv = _predict_panel(mdl, X, val_dates, uni, cfg)
                ic = _daily_spearman_ic(pv, y.loc[val_dates], min_names=cfg.min_names)
                mu, sd, ir = _ic_stats(ic)
                diag[name] = {"ic": ic, "mean_ic": mu, "std_ic": sd, "ir": ir}

            w_ens = _ensemble_weights_from_val(diag)
            t_refit = dates[rp]
            ensemble_hist[t_refit] = w_ens
            ic_hist[t_refit] = {k: (v["mean_ic"], v["ir"]) for k, v in diag.items()}

            # Refit on train + validation combined, then predict the next block
            fit_dates = dates[(dates >= tr0) & (dates <= va1)]
            Xfit, yfit, _, _ = _panel_to_samples(X, y, fit_dates, uni, cfg)

            if Xfit.shape[0] == 0:
                # print(f"  [!] Skipping forward fit at {t_refit.date()}: No data.")
                continue

            preds_fwd = {}
            nxt = min(len(dates), rp + cfg.refit_every)
            block_dates = dates[rp:nxt]
            for name, mdl in models.items():
                mdl.fit(Xfit, yfit)
                preds_fwd[name] = _predict_panel(mdl, X, block_dates, uni, cfg)

            pred_block = _combine_predictions(preds_fwd, w_ens)
            pred_all.loc[pred_block.index] = pred_block

        # Step 5: Post-process alpha
        alpha = _cs_robust(pred_all, cfg.winsor_k, cfg.gauss_cap)
        alpha_n = _cs_neutralize_panel(alpha, sectors=sectors, beta=beta,
                                        mask=uni, min_names=cfg.min_names)

        # Store diagnostics and intermediate objects for the portfolio constructor
        self.diagnostics_ = {
            "ensemble_weights_by_refit": ensemble_hist,
            "val_ic_by_refit": ic_hist,
        }
        self.uni_ = uni
        self.beta_ = beta
        self.pos_cap_, self.trade_cap_ = _position_caps_usd(prices, volume, cfg)
        self.sectors_ = sectors
        self.prices_ = prices

        return alpha_n
    

from sklearn.linear_model import RidgeCV

import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from tqdm import tqdm

import pandas as pd
import numpy as np

class ShortTermSignalGenerator:
    """Generates fast, mean-reversion signals based on short-term price action."""
    def __init__(self, reversal_window=5, smoothing_span=3):
        self.window = reversal_window
        self.span = smoothing_span

    def generate(self, hedged_returns):
        print(f"  -> Generating Short-Term Signals ({self.window}d Reversal)...")
        # 1. Calculate short-term returns
        ret_short = hedged_returns.rolling(self.window, min_periods=self.window-2).sum()
        
        # 2. Cross-sectional Z-score
        cs_mean = ret_short.mean(axis=1)
        cs_std = ret_short.std(axis=1)
        z_score = ret_short.sub(cs_mean, axis=0).div(cs_std + 1e-8, axis=0)
        
        # 3. Invert for Mean Reversion (Buy losers, sell winners)
        signal = -z_score
        
        # 4. Light smoothing to prevent daily whipsaw
        return signal.ewm(span=self.span, min_periods=1).mean()

import pandas as pd
import numpy as np

class ShortTermSignalGenerator:
    """Generates fast, mean-reversion signals based on short-term price action."""
    def __init__(self, reversal_window=5, smoothing_span=3):
        self.window = reversal_window
        self.span = smoothing_span

    def generate(self, hedged_returns):
        print(f"  -> Generating Short-Term Signals ({self.window}d Reversal)...")
        # 1. Calculate short-term returns
        ret_short = hedged_returns.rolling(self.window, min_periods=self.window-2).sum()
        
        # 2. Cross-sectional Z-score
        cs_mean = ret_short.mean(axis=1)
        cs_std = ret_short.std(axis=1)
        z_score = ret_short.sub(cs_mean, axis=0).div(cs_std + 1e-8, axis=0)
        
        # 3. Invert for Mean Reversion (Buy losers, sell winners)
        signal = -z_score
        
        # 4. Light smoothing to prevent daily whipsaw
        return signal.ewm(span=self.span, min_periods=1).mean()


class LongTermSignalGenerator:
    """
    Generates long-term signals using Momentum as the primary engine, 
    but strictly uses Sector-Neutral Value as a 'Guardrail' multiplier to 
    penalize expensive bubbles and reward cheap compounders.
    """
    def __init__(self, momentum_window=252, skip_recent=21, smoothing_span=10, value_tilt_strength=0.25):
        self.window = momentum_window
        self.skip = skip_recent
        self.span = smoothing_span
        self.tilt = value_tilt_strength # Controls how much Value influences Momentum

    def generate(self, hedged_returns, earnings_yield, sectors):
        print(f"  -> Generating Long-Term Signals (Value-Tilted Momentum)...")
        eps = 1e-8
        dates = hedged_returns.index
        tickers = hedged_returns.columns
        
        # --- 1. Momentum Component (The Core Engine) ---
        ret_long = hedged_returns.rolling(self.window - self.skip).sum().shift(self.skip)
        mom_z = ret_long.sub(ret_long.mean(axis=1), axis=0).div(ret_long.std(axis=1) + eps, axis=0).fillna(0)
        
        # --- 2. Value Component (The Guardrail) ---
        ey_aligned = earnings_yield.ffill().fillna(0)
        val_z = pd.DataFrame(0.0, index=dates, columns=tickers)
        
        unique_sectors = sectors.unique()
        for sec in unique_sectors:
            sec_tickers = sectors[sectors == sec].index.intersection(tickers)
            if len(sec_tickers) > 1:
                sec_ey = ey_aligned[sec_tickers]
                # Z-score Sector Neutral EY
                sec_z = sec_ey.sub(sec_ey.mean(axis=1), axis=0).div(sec_ey.std(axis=1) + eps, axis=0)
                val_z[sec_tickers] = sec_z
                
        val_z = val_z.fillna(0)
        
        # --- 3. The Conviction Multiplier (Robust Blend) ---
        # Clip value to strictly prevent extreme data outliers from breaking the signal
        safe_val = val_z.clip(lower=-2.0, upper=2.0)
        
        # Create a multiplier: If tilt is 0.25, multiplier ranges from 0.5x to 1.5x
        # Cheap stock = Multiplier > 1 (Boost signal)
        # Expensive stock = Multiplier < 1 (Shrink signal)
        value_multiplier = 1.0 + (safe_val * self.tilt)
        
        # Apply the tilt to the Momentum signal
        tilted_mom = mom_z * value_multiplier
        
        # --- 4. Re-Normalize and Smooth ---
        cs_mean = tilted_mom.mean(axis=1)
        cs_std = tilted_mom.std(axis=1)
        combined_z = tilted_mom.sub(cs_mean, axis=0).div(cs_std + eps, axis=0)
        
        return combined_z.fillna(0).ewm(span=self.span, min_periods=1).mean()


class DynamicSignalBlender:
    """
    Allocates weight between Short and Long signals using a Composite Regime Model.
    Evaluates both Volatility and Trend Strength independently, and uses a Logistic 
    Sigmoid function to safely map regimes to signal weights.
    """
    def __init__(self, fast_win=21, slow_win=252, trend_win=60):
        self.fast_win = fast_win
        self.slow_win = slow_win
        self.trend_win = trend_win

    def blend(self, short_signals, long_signals, benchmark_returns):
        print("  -> Blending signals using Composite Sigmoid Regime Model...")
        
        # We use a 2-year lookback to baseline what "normal" market conditions are
        baseline_lookback = 252 * 2 
        
        # --- 1. Volatility Regime (The 'Panic' Axis) ---
        bench_vol_fast = benchmark_returns.rolling(self.fast_win, min_periods=10).std()
        bench_vol_slow = benchmark_returns.rolling(self.slow_win, min_periods=60).std()
        
        # Ratio of short-term to long-term vol
        vol_ratio = (bench_vol_fast / (bench_vol_slow + 1e-8)).fillna(1.0)
        
        # Z-score the ratio to detect statistical anomalies 
        vol_z = (vol_ratio - vol_ratio.rolling(baseline_lookback, min_periods=126).mean()) / \
                (vol_ratio.rolling(baseline_lookback, min_periods=126).std() + 1e-8)
                
        # Sigmoid squeeze: High Volatility -> approaches 1.0
        vol_score = 1 / (1 + np.exp(-vol_z.fillna(0)))

        # --- 2. Trend Regime (The 'Directionality' Axis) ---
        # Trend strength = Absolute Return / Realized Volatility over 60 days
        trend_ret = benchmark_returns.rolling(self.trend_win).sum()
        trend_vol = benchmark_returns.rolling(self.trend_win).std() * np.sqrt(self.trend_win)
        trend_strength = (trend_ret / (trend_vol + 1e-8)).abs()
        
        # Z-score the trend strength
        trend_z = (trend_strength - trend_strength.rolling(baseline_lookback, min_periods=126).mean()) / \
                  (trend_strength.rolling(baseline_lookback, min_periods=126).std() + 1e-8)
                  
        # Sigmoid squeeze: Strong Trend -> approaches 1.0
        trend_score = 1 / (1 + np.exp(-trend_z.fillna(0)))

        # --- 3. Dynamic Weight Allocation ---
        # We want Short-Term Mean Reversion when Volatility is HIGH and Trend is LOW.
        # Starting from a 50/50 baseline, we add the vol score and subtract the trend score.
        raw_short_weight = 0.5 + 0.5 * (vol_score - trend_score)
        
        # Strict constraints: Never allocate more than 80% or less than 20% to either strategy
        short_weight = raw_short_weight.clip(lower=0.20, upper=0.80)
        long_weight = 1.0 - short_weight
        
        # --- 4. Apply and Re-Normalize ---
        # Broadcast the 1D weight series across the 2D signal dataframes
        blended = short_signals.mul(short_weight, axis=0) + long_signals.mul(long_weight, axis=0)
        
        # Cross-sectional Z-score so the Portfolio Constructor gets standard inputs
        cs_mean = blended.mean(axis=1)
        cs_std = blended.std(axis=1)
        final_signal = blended.sub(cs_mean, axis=0).div(cs_std + 1e-8, axis=0)
        
        return final_signal

class RollingLinearSignal(SignalGenerator):
    """
    A walk-forward rolling linear model (Ridge Regression).
    
    Fixed to use Time-Series scaling instead of Cross-Sectional neutralization,
    allowing the model to capture directional trends and fat-tailed momentum.
    """
    # Changed default fwd_days to 21 to match the refit window and avoid 1-day noise fitting
    def __init__(self, train_window=252, refit_every=21, fwd_days=21, ridge_alpha=10.0):
        self.train_window = train_window
        self.refit_every = refit_every
        self.fwd_days = fwd_days
        self.ridge_alpha = ridge_alpha
        self.history_coefs = []

    def get_signals(self, hedged_returns, **kwargs):
        log_returns = np.log1p(hedged_returns)
        dates = log_returns.index
        tickers = log_returns.columns
        
        print("1. Computing base factors...")
        mom = log_returns.rolling(252, min_periods=200).sum() - log_returns.rolling(21, min_periods=15).sum()
        rev = -log_returns.rolling(5, min_periods=3).sum()
        vol = -log_returns.rolling(60, min_periods=40).std(ddof=0)
        
        print("2. Applying time-series scaling (Removed _cs_robust)...")
        # Use a tiny epsilon to prevent division by zero on zero-volatility/flat days
        eps = 1e-8
        z_mom = mom / (mom.rolling(252, min_periods=100).std() + eps)
        z_rev = rev / (rev.rolling(252, min_periods=100).std() + eps)
        z_vol = vol / (vol.rolling(252, min_periods=100).std() + eps)
        
        factors = {"mom": z_mom, "rev": z_rev, "vol": z_vol}
        feat_names = list(factors.keys())
        
        print(f"3. Building target variable (Predicting {self.fwd_days}-day returns)...")
        # Target is raw forward returns. We do NOT neutralize the target.
        fwd = log_returns.rolling(self.fwd_days).sum().shift(-self.fwd_days)
        
        out_signal = pd.DataFrame(index=dates, columns=tickers, dtype=float)
        
        # fit_intercept=True allows the model to handle the base market drift
        model = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        
        start_idx = self.train_window + self.fwd_days
        refits = list(range(start_idx, len(dates), self.refit_every))
        
        print(f"4. Running walk-forward linear model ({len(refits)} refits)...")
        
        is_fitted = False 
        
        for rp in tqdm(refits, desc="Rolling Fit"):
            tr1_idx = rp - self.fwd_days # Ensure NO lookahead bias
            tr0_idx = max(0, tr1_idx - self.train_window)
            train_dates = dates[tr0_idx:tr1_idx]
            
            X_train, y_train = [], []
            for t in train_dates:
                Xt = np.column_stack([factors[f].loc[t].values for f in feat_names])
                yt = fwd.loc[t].values
                valid = np.isfinite(Xt).all(axis=1) & np.isfinite(yt)
                if valid.sum() > 0:
                    X_train.append(Xt[valid])
                    y_train.append(yt[valid])
                
            if X_train:
                X_train = np.vstack(X_train)
                y_train = np.concatenate(y_train)
                
                # Fit if we have enough data points
                if len(y_train) >= 100:
                    model.fit(X_train, y_train)
                    is_fitted = True
                    self.history_coefs.append({
                        "date": dates[rp],
                        "mom_weight": model.coef_[0],
                        "rev_weight": model.coef_[1],
                        "vol_weight": model.coef_[2],
                        "intercept": model.intercept_
                    })
            
            if not is_fitted:
                continue
                
            nxt = min(len(dates), rp + self.refit_every)
            block_dates = dates[rp:nxt]
            
            # Predict
            for t in block_dates:
                Xt = np.column_stack([factors[f].loc[t].values for f in feat_names])
                valid = np.isfinite(Xt).all(axis=1)
                if valid.sum() > 0:
                    preds = model.predict(Xt[valid])
                    out_signal.loc[t, tickers[valid]] = preds

        # 5. Translate raw return predictions into cross-sectional Z-scores
        # This gives the portfolio engine the relative sizing it expects (mean=0, std=1)
        print("5. Formatting final signals for portfolio constructor...")
        
        # Calculate daily cross-sectional mean and standard deviation
        cs_mean = out_signal.mean(axis=1)
        cs_std = out_signal.std(axis=1)
        
        # Z-score the signals day-by-day (using a tiny epsilon to prevent division by zero)
        final_signal = out_signal.sub(cs_mean, axis=0).div(cs_std + 1e-8, axis=0)
        
        return final_signal

class TimeSeriesMultiFactor(SignalGenerator):
    """
    Combines factors using historical time-series scaling rather than 
    daily cross-sectional standardizing. This preserves outliers and 
    overall market directionality.
    """
    def __init__(self, mom_weight=1.0, rev_weight=0.5, vol_weight=0.5):
        # Slightly down-weighting rev and vol to let momentum lead
        self.mom_weight = mom_weight
        self.rev_weight = rev_weight
        self.vol_weight = vol_weight

    def get_signals(self, hedged_returns, **kwargs):
        log_returns = np.log1p(hedged_returns)
        
        # 1. Base factors
        mom = log_returns.rolling(252, min_periods=200).sum() - log_returns.rolling(21, min_periods=15).sum()
        rev = -log_returns.rolling(5, min_periods=3).sum()
        vol = -log_returns.rolling(60, min_periods=40).std(ddof=0)
        
        # 2. Time-Series Scaling (NOT Cross-Sectional)
        # We divide by the rolling 252-day standard deviation of the FACTOR ITSELF 
        # to normalize the scale, but without forcing the cross-section to sum to zero.
        mom_scaled = mom / mom.rolling(252, min_periods=100).std()
        rev_scaled = rev / rev.rolling(252, min_periods=100).std()
        vol_scaled = vol / vol.rolling(252, min_periods=100).std()
        
        # 3. Combine without clipping
        signal = (
            (mom_scaled * self.mom_weight) + 
            (rev_scaled * self.rev_weight) + 
            (vol_scaled * self.vol_weight)
        )
        
        # Return the raw combined signal. No _cs_robust!
        return signal