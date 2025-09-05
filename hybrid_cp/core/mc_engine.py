# mc_engine.py  (v2-fixed)
import os
import re
import warnings
from datetime import timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import GraphicalLassoCV
from sklearn.exceptions import ConvergenceWarning

SUMMER_MONTHS = [6, 7, 8, 9]

# ---------- Robust timestamp parsing (handles 24:.., missing seconds, zero-width spaces) ----------
_iso_pat = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def _fallback_monthly_thresholds(date, k, daily_actual, hourly_actual):
    y, m = date.year, date.month
    hist_hour = hourly_actual[(hourly_actual.index.month == m) & (hourly_actual.index < date)]
    hist_day = daily_actual[(daily_actual.index.month == m) & (daily_actual.index < date)]

    def _topk_or_p95(series, k):
        vals = np.asarray(series, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size >= k:
            return float(np.sort(vals)[-k])
        if vals.size >= 10:
            return float(np.quantile(vals, 0.95))
        return np.nan

    Dk = _topk_or_p95(hist_hour['hourly_max'].values if len(hist_hour) else np.array([]), k)
    Mk = _topk_or_p95(hist_day['actual_mw'].values if len(hist_day) else np.array([]), k)

    if not np.isfinite(Dk):
        all_hour = hourly_actual[hourly_actual.index < date]
        if len(all_hour):
            Dk = float(np.quantile(all_hour['hourly_max'].values, 0.95))
    if not np.isfinite(Mk):
        all_day = daily_actual[daily_actual.index < date]
        if len(all_day):
            Mk = float(np.quantile(all_day['actual_mw'].values, 0.95))

    return Dk, Mk


def _parse_timestamp_series(date_s: pd.Series, time_s: pd.Series) -> pd.Series:
    """Parse Date/Time robustly:
       - 支持 H:MM / HH:MM / HH:MM:SS
       - 24:.. → 次日 00:..
       - 对 YYYY-MM-DD 用显式格式消除 dayfirst 告警
       - 返回 dtype=datetime64[ns] 的 Series
    """

    def _parse_date_only(d: str):
        d = d.strip().replace('\u200b', '')
        if _iso_pat.match(d):
            # 明确格式，避免 dayfirst 告警
            return pd.to_datetime(d, format="%Y-%m-%d", errors='coerce')
        return pd.to_datetime(d, dayfirst=True, errors='coerce')

    out = []
    for d, t in zip(date_s, time_s):
        d = str(d).strip().replace('\u200b', '')
        t = str(t).strip().replace('\u200b', '')
        parts = t.split(':')
        if not parts or not parts[0].isdigit():
            out.append(pd.NaT);
            continue
        try:
            hh = int(parts[0])
            mm = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0
            ss = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 0
        except Exception:
            out.append(pd.NaT);
            continue

        base = _parse_date_only(d)
        if pd.isna(base):
            out.append(pd.NaT);
            continue

        if hh == 24:
            base = base + timedelta(days=1)
            hh = 0

        if 0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59:
            out.append(pd.Timestamp(base.year, base.month, base.day, hh, mm, ss))
        else:
            out.append(pd.NaT)

    return pd.Series(out, dtype="datetime64[ns]")


# ---------- Safe hourly regularization ----------
def hourly_regularize_safe(df: pd.DataFrame) -> pd.DataFrame:
    out_parts = []
    for (src, zn), g in df.groupby(['forecast_src', 'zone'], sort=False):
        g = g.sort_values('timestamp')
        g2 = g.set_index('timestamp').asfreq('h')
        # only fill numeric columns
        num_cols = [c for c in g2.columns if pd.api.types.is_numeric_dtype(g2[c])]
        g2[num_cols] = g2[num_cols].ffill().bfill()
        g2 = g2.reset_index()
        g2['forecast_src'] = src
        g2['zone'] = zn
        out_parts.append(g2)
    out = pd.concat(out_parts, ignore_index=True)
    out = out.sort_values('timestamp').reset_index(drop=True)
    return out


# ---------- Base DF builder (per horizon & source) ----------
def build_base_df(csv_path: str, forecast_col: str) -> pd.DataFrame:
    """Load one horizon CSV and construct the base hourly frame."""
    raw = pd.read_csv(csv_path)

    all_cols = raw.columns.tolist()
    fcast_cols = [c for c in all_cols if "Historic Forecast" in c]
    actual_candidates = [c for c in all_cols if "Actual" in c]
    assert forecast_col in fcast_cols, f"forecast_col '{forecast_col}' not in CSV columns {fcast_cols}"
    assert len(actual_candidates) >= 1, "The *Actual* column was not found."
    actual_col = actual_candidates[0]

    # Robust timestamps
    ts = _parse_timestamp_series(raw['Date'], raw['Time'])
    bad = int(pd.isna(ts).sum())
    if bad > 0:
        print(f"[WARN] {os.path.basename(csv_path)}: drop {bad} rows with NaT timestamps.")
        mask = ts.notna()
        raw = raw.loc[mask].reset_index(drop=True)
        ts = ts[mask].reset_index(drop=True)

    df = pd.DataFrame({
        'timestamp': ts,
        'actual_mw': pd.to_numeric(raw[actual_col], errors='coerce').astype(float),
        'da_forecast_mw': pd.to_numeric(raw[forecast_col], errors='coerce').astype(float),
        'forecast_src': forecast_col.split(':')[0].strip(),  # 'ECA' / 'RTO' / 'TESLA'
        'zone': 'Ontario'
    })

    # Merge duplicates and regularize
    df = df.groupby(['forecast_src', 'zone', 'timestamp'], as_index=False) \
        .agg({'actual_mw': 'mean', 'da_forecast_mw': 'mean'})
    df = hourly_regularize_safe(df)
    df = df.dropna(subset=['actual_mw', 'da_forecast_mw']).sort_values('timestamp').reset_index(drop=True)
    return df


# ---------- Hourly features / daily 24h residual matrix ----------
def build_hourly_features(df_base: pd.DataFrame) -> pd.DataFrame:
    x = df_base.copy()
    x["date"] = x["timestamp"].dt.floor("D")
    x["year"] = x["timestamp"].dt.year
    x["month"] = x["timestamp"].dt.month
    x["hour"] = x["timestamp"].dt.hour
    x["residual"] = x["actual_mw"] - x["da_forecast_mw"]
    return x


def make_hourly_matrix(features_hourly: pd.DataFrame) -> pd.DataFrame:
    mat = features_hourly.pivot_table(index="date", columns="hour", values="residual", aggfunc="mean")
    for h in range(24):
        if h not in mat.columns:
            mat[h] = np.nan
    mat = mat[[h for h in range(24)]].sort_index()
    mat.columns = [f"res_h{h:02d}" for h in range(24)]
    return mat.reset_index()


# ---------- POT marginal + Gaussianization ----------
class MarginalPOT:
    def __init__(self, hour: int, p_u: float, u: float, xi: float, beta: float, center_vals: np.ndarray):
        self.hour = hour
        self.p_u = p_u
        self.u = u
        self.xi = xi
        self.beta = beta
        self.center_vals = np.asarray(center_vals, dtype=float)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        u = np.empty_like(x)

        mask_body = x <= self.u
        if mask_body.any():
            vals = self.center_vals
            ranks = np.searchsorted(vals, x[mask_body], side="right")
            h = np.clip(ranks / max(len(vals), 1), 0.0, 1.0)
            u[mask_body] = self.p_u * h

        mask_tail = ~mask_body
        if mask_tail.any():
            z = (x[mask_tail] - self.u) / max(self.beta, 1e-8)
            if abs(self.xi) < 1e-6:
                g = 1 - np.exp(-z)
            else:
                g = 1 - np.power(np.maximum(1 + self.xi * z, 1e-9), -1 / self.xi)
            u[mask_tail] = self.p_u + (1 - self.p_u) * g

        return np.clip(u, 1e-9, 1 - 1e-9)

    def to_normal(self, x: np.ndarray) -> np.ndarray:
        u = self.cdf(x)
        return stats.norm.ppf(u)


def fit_marginals_pot(hourly_mat: pd.DataFrame, p_u: float = 0.95) -> Dict[int, MarginalPOT]:
    out = {}
    for h in range(24):
        vals = hourly_mat[f"res_h{h:02d}"].dropna().values.astype(float)
        if len(vals) == 0:
            out[h] = MarginalPOT(h, p_u, 0.0, 0.0, 1.0, center_vals=np.array([0.0]))
            continue
        vals_sorted = np.sort(vals)
        u = np.quantile(vals_sorted, p_u)
        body = vals_sorted[vals_sorted <= u]
        tail = vals_sorted[vals_sorted > u]
        if len(tail) < 10:
            xi, beta = 0.1, np.std(vals_sorted) + 1e-6
        else:
            excess = tail - u
            shape, loc, scale = stats.genpareto.fit(excess, floc=0.0)  # xi, 0, beta
            xi, beta = float(shape), float(scale)
        out[h] = MarginalPOT(h, p_u=p_u, u=float(u), xi=float(xi), beta=float(beta),
                             center_vals=body if len(body) else vals_sorted)
    return out


def gaussianize(hourly_mat: pd.DataFrame, transforms: Dict[int, MarginalPOT]) -> np.ndarray:
    Z = np.zeros((len(hourly_mat), 24))
    it = hourly_mat.itertuples(index=False)
    for i, row in enumerate(it):
        for h in range(24):
            v = getattr(row, f"res_h{h:02d}")
            if pd.isna(v):
                Z[i, h] = 0.0
            else:
                Z[i, h] = transforms[h].to_normal(np.array([float(v)]))[0]
    return Z


# ---------- Scenario generation + monthly Top-4 thresholds ----------
def scenario_probabilities_summer_topk(
        df_base: pd.DataFrame,
        n_sims: int = 2000,
        k: int = 4,
        return_peak_hour: bool = False
) -> pd.DataFrame:
    # Residuals
    feats_hourly = build_hourly_features(df_base)
    hourly_mat = make_hourly_matrix(feats_hourly)

    # Gaussianize
    transforms = fit_marginals_pot(hourly_mat, p_u=0.95)
    Z = gaussianize(hourly_mat, transforms)

    # Graphical Lasso CV with stronger convergence and fallback
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        gl = GraphicalLassoCV(max_iter=500)
        cov = None
        try:
            gl.fit(Z)
            cov = gl.covariance_
        except Exception:
            cov = None

    if cov is None or not np.isfinite(cov).all():
        emp = np.cov(Z, rowvar=False)
        cov = emp + 1e-6 * np.eye(emp.shape[0])

    mean = np.nanmean(Z, axis=0)

    # Forecast 24h matrix
    x = df_base.copy()
    x['date'] = x['timestamp'].dt.floor('D')
    x['hour'] = x['timestamp'].dt.hour
    fmat = x.pivot_table(index='date', columns='hour', values='da_forecast_mw', aggfunc='mean')
    for h in range(24):
        if h not in fmat.columns: fmat[h] = np.nan
    fmat = fmat[[h for h in range(24)]].sort_index()
    fmat.columns = [f"fh{h:02d}" for h in range(24)]

    # Running thresholds till yesterday (same month)
    base = df_base.copy()
    base['date'] = base['timestamp'].dt.floor('D')
    base['year'] = base['timestamp'].dt.year
    base['month'] = base['timestamp'].dt.month
    daily_actual = base.groupby('date', as_index=False)['actual_mw'].max().set_index('date').sort_index()
    hourly_actual = base.groupby(['date'], as_index=False).agg(hourly_max=('actual_mw', 'max')).set_index(
        'date').sort_index()

    out = []
    dates = fmat.index
    rng = np.random.default_rng(42)

    for date in dates:
        y, m = date.year, date.month
        if m not in SUMMER_MONTHS:
            out.append({"date": date,
                        "p_4cp_hourly_monthlyTop4": np.nan,
                        "p_4cp_daily_monthlyTop4": np.nan,
                        "p_4cp_hourly_mlh": np.nan})
            continue

        mask_h = (hourly_actual.index.year == y) & (hourly_actual.index.month == m) & (hourly_actual.index < date)
        mask_d = (daily_actual.index.year == y) & (daily_actual.index.month == m) & (daily_actual.index < date)
        prev_hourly = hourly_actual.loc[mask_h]
        prev_daily = daily_actual.loc[mask_d]

        def _topk_or_p95(arr, k):
            arr = np.asarray(arr, float)
            arr = arr[np.isfinite(arr)]
            if arr.size >= k:       return float(np.sort(arr)[-k])
            if arr.size >= 10:      return float(np.quantile(arr, 0.95))
            return np.nan

        if len(prev_hourly) >= k and len(prev_daily) >= k:
            Dk_prev = float(np.sort(prev_hourly['hourly_max'].values)[-k])
            Mk_prev = float(np.sort(prev_daily['actual_mw'].values)[-k])
        else:
            hist_hour = hourly_actual[(hourly_actual.index.month == m) & (hourly_actual.index < date)]
            hist_day = daily_actual[(daily_actual.index.month == m) & (daily_actual.index < date)]
            Dk_prev = _topk_or_p95(hist_hour['hourly_max'].values if len(hist_hour) else [], k)
            Mk_prev = _topk_or_p95(hist_day['actual_mw'].values if len(hist_day) else [], k)
            if not np.isfinite(Dk_prev):
                all_hour = hourly_actual[hourly_actual.index < date]
                if len(all_hour): Dk_prev = float(np.quantile(all_hour['hourly_max'].values, 0.95))
            if not np.isfinite(Mk_prev):
                all_day = daily_actual[daily_actual.index < date]
                if len(all_day):  Mk_prev = float(np.quantile(all_day['actual_mw'].values, 0.95))
            if not (np.isfinite(Dk_prev) and np.isfinite(Mk_prev)):
                out.append({"date": date,
                            "p_4cp_hourly_monthlyTop4": np.nan,
                            "p_4cp_daily_monthlyTop4": np.nan,
                            "p_4cp_hourly_mlh": np.nan})
                continue

        sims = rng.multivariate_normal(mean, cov, size=int(n_sims))
        resims = np.zeros_like(sims)
        for h in range(24):
            vals = hourly_mat[f"res_h{h:02d}"].dropna().values.astype(float)
            if len(vals) == 0:
                resims[:, h] = 0.0
            else:
                vals_sorted = np.sort(vals)
                u = stats.norm.cdf(sims[:, h])
                ranks = np.clip((len(vals_sorted) - 1) * u, 0, len(vals_sorted) - 1).astype(int)
                resims[:, h] = vals_sorted[ranks]

        fh = fmat.loc[date].values.astype(float) if date in fmat.index else np.zeros(24, dtype=float)
        fh = np.where(np.isfinite(fh), fh, 0.0)
        scen = resims + fh  # (n_sims, 24)

        sim_hourly_max = scen.max(axis=1)
        sim_daily_max = scen.max(axis=1)
        p4h = float((sim_hourly_max > Dk_prev).mean())
        p4d = float((sim_daily_max > Mk_prev).mean())

        try:
            peak_idx = np.nanargmax(scen, axis=1)
        except ValueError:
            peak_idx = np.argmax(np.nan_to_num(scen, nan=-1e18), axis=1)
        ph = np.bincount(peak_idx, minlength=24) / scen.shape[0]
        exceed_rate_by_h = np.nanmean((scen > Dk_prev).astype(float), axis=0)
        p4h_mlh = float(np.dot(ph, exceed_rate_by_h))

        rec = {"date": date,
               "p_4cp_hourly_monthlyTop4": p4h,
               "p_4cp_daily_monthlyTop4": p4d,
               "p_4cp_hourly_mlh": p4h_mlh}
        if return_peak_hour:
            rec.update({f"ph_{h:02d}": float(ph[h]) for h in range(24)})
            rec["peak_hour_mlh"] = int(np.argmax(ph))
        out.append(rec)

    prob = pd.DataFrame(out).set_index('date').sort_index()
    return prob


# ---------- Confusion helper ----------
def evaluate_confusion_4cp(prob_df: pd.DataFrame, labels_df: pd.DataFrame, kind: str) -> Tuple[int, int, int, int]:
    df = pd.DataFrame({'p': prob_df[kind]}).copy()
    df = df.rename_axis('date').reset_index()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    lab = labels_df[['timestamp', 'is_CP']].copy()
    lab['date'] = pd.to_datetime(lab['timestamp']).dt.floor('D')

    d = pd.merge(df[['date', 'p', 'month']], lab[['date', 'is_CP']], on='date', how='left')
    d = d.dropna(subset=['p', 'is_CP']).copy()
    d = d[d['month'].isin(SUMMER_MONTHS)].copy()

    y_true = d['is_CP'].astype(int).values
    y_pred = (d['p'].values >= 0.5).astype(int)

    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    return TP, FP, TN, FN
