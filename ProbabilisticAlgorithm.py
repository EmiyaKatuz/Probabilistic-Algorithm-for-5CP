import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta


# Probabilistic Algorithm for 5CP
def Probabilistic(thres_d, thres_p, actualdemandfile, forecastdemandfile):
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

    calledCP = []  # to store called 5CP days
    calledCP_with_date = []
    delta = [271, 210, 653, 380, 545, 584]  # normal distribution delta

    for i in range(1, N):
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
        if month < 5 or month > 9:
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
        # d_tomorrow = demand_forecast[((i - 1) * 6), 3]  # tomorrow's peak forecast on day i
        # d_forecast = demand_forecast[((i - 1) * 6 + 1):((i - 1) * 6 + 6), 3]  # short term peak forecast on day i

        # d_past_seasonal = []
        # for row in demand_actual:
        #     date_str = str(int(row[0]))
        #     if len(date_str) == 8:
        #         year = int(date_str[:4])
        #         month = int(date_str[4:6])
        #         if 5 <= month <= 9 and (year == 2022 or year ==2021):
        #             d_past_seasonal.append(row[1])
        # d_past = np.array(d_past_seasonal)

        d_past = demand_actual[:i + 365, 1]

        # t_tomorrow = weatherdata[i - 1, 2]  # tomorrow's temperature forecast for the peak hour

        if d_tomorrow >= thres_d:
            # Compute Probabilities where P_future[i] represent the probability of tomorrow's peak ranking (i+1)_th in forecast data, i=0 to 4
            P_future = np.zeros(5)
            norm_p = np.zeros(
                5)  # store P(tomorrow's demand>=forecast demand for day(tomorrow+j+1) for 6 days short term forecast, excluding tomorrow
            for j in range(0, 5):
                norm_p[j] = norm.cdf(d_tomorrow - d_forecast[j], 0, delta[j + 1] + delta[0])
            P_future[0] = 1
            for j in range(0, 5):
                P_future[0] = P_future[0] * norm_p[j]
            P_future[1] = norm_p[0] * norm_p[1] * norm_p[2] * norm_p[3] * (1 - norm_p[4]) + norm_p[0] * norm_p[1] * \
                          norm_p[2] * norm_p[4] * (1 - norm_p[3]) + norm_p[0] * norm_p[1] * norm_p[4] * norm_p[3] * (
                                  1 - norm_p[2]) + norm_p[0] * norm_p[4] * norm_p[2] * norm_p[3] * (1 - norm_p[1]) + \
                          norm_p[4] * norm_p[1] * norm_p[2] * norm_p[3] * (1 - norm_p[0])
            P_future[2] = norm_p[0] * norm_p[1] * norm_p[2] * (1 - norm_p[3]) * (1 - norm_p[4]) + norm_p[0] * norm_p[
                1] * norm_p[3] * (1 - norm_p[2]) * (1 - norm_p[4]) + norm_p[0] * norm_p[1] * norm_p[4] * (
                                  1 - norm_p[3]) * (1 - norm_p[2]) + norm_p[0] * norm_p[2] * norm_p[3] * (
                                  1 - norm_p[1]) * (1 - norm_p[4]) + norm_p[0] * norm_p[2] * norm_p[4] * (
                                  1 - norm_p[3]) * (1 - norm_p[1]) + norm_p[0] * norm_p[3] * norm_p[4] * (
                                  1 - norm_p[1]) * (1 - norm_p[2]) + norm_p[1] * norm_p[2] * norm_p[3] * (
                                  1 - norm_p[0]) * (1 - norm_p[4]) + norm_p[1] * norm_p[2] * norm_p[4] * (
                                  1 - norm_p[0]) * (1 - norm_p[3]) + norm_p[1] * norm_p[3] * norm_p[4] * (
                                  1 - norm_p[0]) * (1 - norm_p[2]) + norm_p[2] * norm_p[3] * norm_p[4] * (
                                  1 - norm_p[0]) * (1 - norm_p[1])
            P_future[3] = norm_p[0] * norm_p[1] * (1 - norm_p[2]) * (1 - norm_p[3]) * (1 - norm_p[4]) + norm_p[0] * \
                          norm_p[2] * (1 - norm_p[1]) * (1 - norm_p[3]) * (1 - norm_p[4]) + norm_p[0] * norm_p[3] * (
                                  1 - norm_p[2]) * (1 - norm_p[1]) * (1 - norm_p[4]) + norm_p[0] * norm_p[4] * (
                                  1 - norm_p[2]) * (1 - norm_p[3]) * (1 - norm_p[1]) + norm_p[1] * norm_p[2] * (
                                  1 - norm_p[0]) * (1 - norm_p[3]) * (1 - norm_p[4]) + norm_p[1] * norm_p[3] * (
                                  1 - norm_p[2]) * (1 - norm_p[0]) * (1 - norm_p[4]) + norm_p[1] * norm_p[4] * (
                                  1 - norm_p[2]) * (1 - norm_p[3]) * (1 - norm_p[0]) + norm_p[2] * norm_p[3] * (
                                  1 - norm_p[0]) * (1 - norm_p[4]) * (1 - norm_p[1]) + norm_p[2] * norm_p[4] * (
                                  1 - norm_p[0]) * (1 - norm_p[3]) * (1 - norm_p[1]) + norm_p[3] * norm_p[4] * (
                                  1 - norm_p[2]) * (1 - norm_p[0]) * (1 - norm_p[1])
            P_future[4] = norm_p[0] * (1 - norm_p[1]) * (1 - norm_p[2]) * (1 - norm_p[3]) * (1 - norm_p[4]) + norm_p[
                1] * (1 - norm_p[0]) * (1 - norm_p[2]) * (1 - norm_p[3]) * (1 - norm_p[4]) + norm_p[2] * (
                                  1 - norm_p[1]) * (1 - norm_p[0]) * (1 - norm_p[3]) * (1 - norm_p[4]) + norm_p[
                              3] * (1 - norm_p[1]) * (1 - norm_p[2]) * (1 - norm_p[0]) * (1 - norm_p[4]) + norm_p[4] * (
                                  1 - norm_p[1]) * (1 - norm_p[2]) * (1 - norm_p[3]) * (1 - norm_p[0])

            # Compute Probabilities where P_past[i] represent the probability of tomorrow's peak ranking (i+1)_th in historic data, i=0 to 4
            P_past = np.zeros(5)
            d_past = d_past[~np.isnan(d_past)]
            d_past.sort()
            d_past_top = d_past[-6:-1]
            P_past[0] = 1 - norm.cdf(d_past_top[4] - d_tomorrow, 0,
                                     delta[0])  # 1-P(tomorrow's peak<=top 1 in historic peaks)
            for j in range(1, 5):
                P_past[j] = norm.cdf(d_past_top[5 - j] - d_tomorrow, 0, delta[0]) - norm.cdf(
                    d_past_top[4 - j] - d_tomorrow, 0, delta[0])

            # Compute Probabilities where P_overall[i] represent the probability of tomorrow's peak ranking (i+1)_th in all data, i=0 to 4
            P_overall = np.zeros(5)
            P_overall[0] = P_future[0] * P_past[0]
            P_overall[1] = P_future[0] * P_past[1] + P_future[1] * P_past[0]
            P_overall[2] = P_future[0] * P_past[2] + P_future[1] * P_past[1] + P_future[2] * P_past[0]
            P_overall[3] = P_future[0] * P_past[3] + P_future[1] * P_past[2] + P_future[2] * P_past[1] + P_future[3] * \
                           P_past[0]
            P_overall[4] = P_future[0] * P_past[4] + P_future[1] * P_past[3] + P_future[2] * P_past[2] + P_future[3] * \
                           P_past[1] + P_future[4] * P_past[0]
            # if the P_overall(rank<=5) is greater than the probability threshold, tomorrow is alerted as a 5CP day.

            if sum(P_overall) >= thres_p:
                # call as a peak day
                calledCP.append(i)
                if i in date_map:
                    calledCP_with_date.append((i, current_forecast_date))
        # Update demand threshold
        elif d_tomorrow < thres_d:
            thres_d = d_tomorrow
    # Output the called peak days
    print("Called CP:", calledCP)
    print("Called CP with date:", calledCP_with_date)


if __name__ == "__main__":
    Probabilistic(15000, 0.1, 'ActualDemand_new.csv', 'ForecastPeak_RTO_new.csv')
