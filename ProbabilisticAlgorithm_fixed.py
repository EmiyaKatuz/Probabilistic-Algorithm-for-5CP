import numpy as np
from scipy.stats import multivariate_normal
from datetime import datetime
from collections import defaultdict

year_probs = defaultdict(list)
hist_pred = []


def rank_probabilities_montecarlo(
        d_tomorrow,
        d_forecast,
        sigma0, sigmas_future,
        past_peaks,
        K=5,
        n_sim=10000,
        rho=0.5
):
    # -------- Constructing the covariance matrix Σ (6×6) -----------------
    Sigma = np.zeros((6, 6))
    Sigma[0, 0] = sigma0 ** 2
    for i in range(5):
        Sigma[i + 1, i + 1] = sigmas_future[i] ** 2
        # The correlation with tomorrow's forecast can be set separately or considered independent.
        Sigma[0, i + 1] = Sigma[i + 1, 0] = 0.0
        # The next 5 days are correlated (empirically) in pairs, using rho
        for j in range(i + 1, 5):
            Sigma[i + 1, j + 1] = Sigma[j + 1, i + 1] = rho * sigmas_future[i] * sigmas_future[j]

    mu = np.concatenate(([d_tomorrow], d_forecast))

    # -------- Monte-Carlo sample -----------------------
    samples = multivariate_normal.rvs(mean=mu, cov=Sigma, size=n_sim)
    E0 = samples[:, 0]  # shape=(n_sim,)
    Efut = samples[:, 1:]  # shape=(n_sim,5)

    # Take the last "most extreme" 5 peaks, not the "last" 5 records.
    if len(past_peaks) >= 5:
        combined_peaks = np.sort(past_peaks)[-5:]  # Top-5
    else:
        combined_peaks = past_peaks  # If less than 5, use them all

    rank_counts = np.zeros(K, dtype=int)

    for s in range(n_sim):
        # Combined array: past constant + 5 samples in the future
        others = np.concatenate([combined_peaks, Efut[s]])
        rank = 1 + (others > E0[s]).sum()  # ranking 1 maximum
        if rank <= K:
            rank_counts[rank - 1] += 1

    return rank_counts / n_sim


# Probabilistic Algorithm for 5CP
def Probabilistic(actualdemandfile, forecastdemandfile):
    # Sample input
    np.set_printoptions(suppress=True)

    # weatherdata = np.genfromtxt(weatherfile, delimiter=',')
    demand_actual = np.genfromtxt(actualdemandfile, delimiter=',')
    demand_forecast = np.genfromtxt(forecastdemandfile, delimiter=',')
    print("forecastdata shape:", demand_forecast.shape)
    print("first 6 rows:\n", demand_forecast[:6])
    print("actualdata shape:", demand_actual.shape)
    print("first 6 rows:\n", demand_actual[:6])

    forecast_dates = set()
    for row in demand_forecast:
        forecast_date = int(row[0])
        forecast_dates.add(forecast_date)

    num_forecast_dates = len(forecast_dates)
    print(f"Number of unique forecast dates: {num_forecast_dates}")

    N = num_forecast_dates + 1
    print(f"Setting N = {N}")

    date_map = {}
    sorted_dates = sorted(list(forecast_dates))
    for idx, date in enumerate(sorted_dates, 1):
        date_map[idx] = date

    delta = [271, 210, 653, 380, 545, 584]  # normal distribution delta

    for i in range(1, N):
        if (len(hist_pred) == 0) or (hist_pred and hist_pred[-1] // 10000 != this_year):
            hist_pred.clear()

        if i not in date_map:
            print(f"Warning: Index {i} not found in date_map, skipping...")
            continue

        current_forecast_date = date_map[i]

        current_date_str = str(current_forecast_date)
        year = int(current_date_str[:4])
        month = int(current_date_str[4:6])
        day = int(current_date_str[6:8])
        current_date = datetime(year, month, day)

        # seasonal check
        if month < 6 or month > 9:
            # print(f"Day {i} (date: {current_forecast_date}) is outside peak season (May-Sep), skipping...")
            continue

        if current_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
            # print(f"Day {i} (date: {current_forecast_date}) is a weekend, skipping...")
            continue

        forecast_rows = []
        for row_idx, row in enumerate(demand_forecast):
            if int(row[0]) == current_forecast_date:
                forecast_rows.append(row)

        if len(forecast_rows) < 6:
            print(f"Warning: Not enough forecast rows for date {current_forecast_date}, skipping...")
            continue

        forecast_rows.sort(key=lambda x: x[2])
        d_tomorrow = forecast_rows[0][3]
        d_forecast = np.array([row[3] for row in forecast_rows[1:6]])

        mask_before_today = demand_actual[:, 0] < current_forecast_date
        months = (demand_actual[:, 0] // 100) % 100
        mask_season = (months >= 6) & (months <= 9)
        this_year = current_forecast_date // 10000
        years = demand_actual[:, 0] // 10000
        mask_this_year = years == this_year
        use_prev_year = ((mask_before_today & mask_this_year & mask_season).sum() < 30)
        mask_year = mask_this_year | ((years == (this_year - 1)) if use_prev_year else False)
        d_past = demand_actual[mask_before_today & mask_year & mask_season, 1]
        if d_past.size < 1:
            continue
        # t_tomorrow = weatherdata[i - 1, 2]  # tomorrow's temperature forecast for the peak hour
        hist_pred.append(d_tomorrow)

        if len(hist_pred) >= 10:
            T_dyn = np.percentile(hist_pred, 95)
        else:
            T_dyn = 0

        if d_tomorrow >= T_dyn:
            sigma0 = delta[0]  # tomorrow's residuals σ
            sigmas_future = delta[1:6]  # residuals for the next 5 days σ
            P_overall = rank_probabilities_montecarlo(
                d_tomorrow, d_forecast,
                sigma0, sigmas_future,
                d_past,  # past actual peak
                K=5,
                n_sim=10000)  # adjustable sampling times
            prob_total = P_overall.sum()
            year_probs[current_date.year].append((current_date, prob_total))
    # Output the called peak days
    for yr, plist in year_probs.items():
        plist_sorted = sorted(plist, key=lambda x: x[1], reverse=True)[:5]
        cp_days = [d.strftime('%Y%m%d') for d, _ in plist_sorted]
        print(f"{yr} Called CP: {cp_days}")


if __name__ == "__main__":
    Probabilistic('ActualDemand_new.csv', 'ForecastPeak_TESLA_new.csv')
