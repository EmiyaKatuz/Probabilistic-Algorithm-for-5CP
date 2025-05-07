#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author : Katuz
@File   : transformation.py
@Project: Probabilistic-Algorithm-for-5CP
@Time   : 2025/4/11 02:06
"""
import pandas as pd

def update_weather_file(infile='weather.csv', outfile='weather_updated.csv'):
    """
    Update weather file:
      - Expected columns (in order): Date, Date+Hour, Temperature
      - Remove header rows (if any) so that only numeric rows remain.
    """
    # Read file and check for header by examining first row values.
    try:
        df = pd.read_csv(infile)
    except Exception as e:
        print(f"Error reading {infile}: {e}")
        return

    # Check if the first column header appears to be a date-label or a number.
    # (If header exists, the dtype will be object instead of numeric.)
    if df.dtypes[0] == 'object':
        print(f"Detected headers in {infile}; removing header in output.")
        # If the file has headers, we assume that the values are non-numeric strings.
        # Remove the header by re-reading with header=0 then writing without headers.
        # (Alternatively, you can directly instruct your code to use skip_header=1.)

    # Write to file without the header (and without the index)
    df.to_csv(outfile, header=False, index=False)
    print(f"Updated weather file saved as: {outfile}")

def update_actual_demand_file(infile='actualdemand.csv', outfile='actualdemand_updated.csv'):
    """
    Update actual demand file:
      - Expected columns (in order): Date, Peak Demand, Peak Hour, Date+Hour
      - Remove header so that numeric data (especially the Peak Demand column) is in column index 1.
    """
    try:
        df = pd.read_csv(infile)
    except Exception as e:
        print(f"Error reading {infile}: {e}")
        return

    if df.dtypes[0] == 'object':
        print(f"Detected headers in {infile}; removing header in output.")

    df.to_csv(outfile, header=False, index=False)
    print(f"Updated actual demand file saved as: {outfile}")

def update_forecast_peak_file(infile='forecastpeak.csv', outfile='forecastpeak_updated.csv'):
    """
    Update forecast peak file:
      - Expected columns (in order): Forecasted Date, Date, DayAhead, Peak Demand Forecast
      - The file must have exactly 6 rows per day (first row is tomorrow's forecast,
        the next five rows are for short-term forecasts).
      - Remove header rows if present.
    """
    try:
        df = pd.read_csv(infile)
    except Exception as e:
        print(f"Error reading {infile}: {e}")
        return

    if df.dtypes[0] == 'object':
        print(f"Detected headers in {infile}; removing header in output.")

    # Optionally, if columns need to be re-ordered (if the CSV’s columns are not in the required order),
    # you could rename/reorder using:
    # df = df[['Forecasted Date', 'Date', 'DayAhead', 'Peak Demand Forecast']]
    # For now, we assume the file’s columns are already in the correct order.

    # Check that the file contains rows in multiples of 6.
    num_rows = len(df)
    if num_rows % 6 != 0:
        print(f"Warning: {infile} has {num_rows} rows which is not a multiple of 6. Please check your data grouping.")
    else:
        print(f"Forecast file row count ({num_rows}) is a multiple of 6.")

    df.to_csv(outfile, header=False, index=False)
    print(f"Updated forecast peak file saved as: {outfile}")

if __name__ == '__main__':
    update_weather_file()
    update_actual_demand_file()
    update_forecast_peak_file()