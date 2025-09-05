import os
import numpy as np
import pandas as pd

SUMMER_MONTHS = [6,7,8,9]

def load_daily_peaks(csv_paths):
    daily = None
    for n, path in sorted(csv_paths.items()):
        df = pd.read_csv(path)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        dt = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
        df['date'] = dt.dt.date
        # Aggregate
        agg = df.groupby('date').agg({
            'ECA: IESO-Ontario Demand Historic Forecast':'max',
            'RTO: IESO-Ontario Demand Historic Forecast':'max',
            'TESLA: IESO-Ontario Demand Historic Forecast':'max',
            'TESLA: IESO-Ontario Demand Actual':'max',
        }).reset_index()
        # Rename forecast columns to horizon-specific names
        agg = agg.rename(columns={
            'ECA: IESO-Ontario Demand Historic Forecast': f'ECA_peak_{n}d',
            'RTO: IESO-Ontario Demand Historic Forecast': f'RTO_peak_{n}d',
            'TESLA: IESO-Ontario Demand Historic Forecast': f'TESLA_pred_peak_{n}d',
            'TESLA: IESO-Ontario Demand Actual': 'actual_peak'
        })
        # Keep a single 'actual_peak' (from the first processed horizon)
        if daily is None:
            daily = agg.copy()
        else:
            agg = agg.drop(columns=['actual_peak'])
            daily = pd.merge(daily, agg, on='date', how='outer')
    # timestamp & calendar
    daily = daily.sort_values('date').reset_index(drop=True)
    daily['timestamp'] = pd.to_datetime(daily['date'])
    daily = daily.drop(columns=['date'])
    daily['year'] = daily['timestamp'].dt.year
    daily['month'] = daily['timestamp'].dt.month
    daily['day'] = daily['timestamp'].dt.day
    daily = daily.dropna(subset=['actual_peak']).reset_index(drop=True)
    return daily

def label_monthly_top4_daily(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df['is_CP'] = 0
    mask = df['month'].between(6,9)
    for (y,m), sub in df[mask].groupby(['year','month']):
        if len(sub) < 20:
            continue
        idx = sub.nlargest(4, 'actual_peak').index
        df.loc[idx, 'is_CP'] = 1
    return df

def build_feature_matrix(daily: pd.DataFrame, selected_feature='ALL'):
    forecast_files = range(1,7)
    feature_cols = []
    if selected_feature == 'ECA':
        feature_cols = [f'ECA_peak_{n}d' for n in forecast_files]
    elif selected_feature == 'RTO':
        feature_cols = [f'RTO_peak_{n}d' for n in forecast_files]
    elif selected_feature == 'TESLA':
        feature_cols = [f'TESLA_pred_peak_{n}d' for n in forecast_files]
    else:
        for base in ['ECA_peak','RTO_peak','TESLA_pred_peak']:
            for n in forecast_files:
                col = f'{base}_{n}d'
                if col in daily.columns:
                    feature_cols.append(col)
    model_df = daily[['timestamp','year','month','actual_peak','is_CP'] + feature_cols].copy()
    return model_df, feature_cols

def split_train_test(model_df: pd.DataFrame, separation_year=2024):
    train_df = model_df[model_df['year'] < separation_year].copy()
    test_df  = model_df[model_df['year'] >= separation_year].copy()
    return train_df, test_df

def select_top_percent_training(model_df: pd.DataFrame, train_df: pd.DataFrame,
                                feature_cols, top_pct=0.10, top_by='actual_peak', scope='overall'):
    scope_df = model_df if scope=='overall' else train_df
    thr = scope_df[top_by].quantile(1.0 - float(top_pct))
    scope_top_idx = scope_df.index[scope_df[top_by] >= thr]
    if scope=='overall':
        train_top_df = model_df.loc[scope_top_idx].copy()
    else:
        train_top_df = train_df.loc[train_df.index.intersection(scope_top_idx)].copy()
    train_top_df = train_top_df.sort_values('timestamp').reset_index(drop=True)
    X_train = train_top_df[feature_cols]
    y_train = train_top_df['is_CP'].astype(int).values
    return X_train, y_train

def summer_mask(df: pd.DataFrame) -> pd.Series:
    return df['month'].isin([6,7,8,9])
