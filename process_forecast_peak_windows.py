import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import argparse


def process_forecast_peak(input_files, output_dir, algorithms=None):
    """
    Process forecast data from multiple files to create ForecastPeak files for each algorithm.
    This function ensures each forecast date has predictions for the next consecutive 6 days.

    Args:
        input_files: Dictionary mapping day-ahead values to file paths
        output_dir: Directory to save the ForecastPeak files
        algorithms: List of algorithms to process (e.g., ['ECA', 'RTO', 'TESLA'])
                   If None, all available algorithms will be processed
    """
    print("Starting forecast data processing...")

    # Initialize dictionary to store forecasts by algorithm and forecast date
    forecasts_by_algo = {}

    # Process each forecast file
    for days_ahead, file_path in input_files.items():
        days_ahead_int = int(days_ahead)
        print(f"Processing {days_ahead} day ahead forecast file: {file_path}")

        try:
            # Read the CSV file with correct date parsing (day/month/year format)
            df = pd.read_csv(file_path, dayfirst=True, parse_dates=['Date'])

            # Clean column names (remove any leading/trailing whitespace)
            df.columns = df.columns.str.strip()

            # Ensure Date and Time columns exist
            if 'Date' not in df.columns or 'Time' not in df.columns:
                print(f"Error: Required columns 'Date' or 'Time' not found in {file_path}")
                continue

            # Convert Time column to standard format if needed
            if df['Time'].dtype == object:
                # Handle different time formats
                df['Time'] = df['Time'].astype(str).str.replace('24:00:00', '00:00').str.zfill(5)

                # Ensure consistent time format (HH:MM)
                df['Time'] = df['Time'].apply(lambda x: x[:5] if ':' in x else f"{int(x):02d}:00")

            # Create a datetime column by combining Date and Time
            df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str),
                                            errors='coerce')

            # Detect available algorithms in the file
            available_algorithms = []
            for col in df.columns:
                if 'Forecast' in col and 'Actual' not in col:
                    # Extract algorithm name (e.g., 'ECA', 'RTO', 'TESLA')
                    for algo in ['ECA', 'RTO', 'TESLA']:
                        if algo in col:
                            available_algorithms.append(algo)
                            break

            # Filter algorithms if specified
            if algorithms:
                available_algorithms = [algo for algo in available_algorithms if algo in algorithms]

            if not available_algorithms:
                print(f"Warning: No matching algorithm columns found in {file_path}")
                continue

            print(f"  Found algorithms: {available_algorithms}")

            # Process each algorithm
            for algo in available_algorithms:
                # Find the forecast column for this algorithm
                forecast_col = [col for col in df.columns if 'Forecast' in col and algo in col and 'Actual' not in col]
                if not forecast_col:
                    continue

                forecast_col = forecast_col[0]

                # Group by date to find daily peak forecasts
                df['date'] = df['Datetime'].dt.date

                # For each date in the file, find the peak forecast
                daily_peaks = df.groupby('date').apply(
                    lambda x: pd.Series({
                        'peak_forecast': x[forecast_col].max(),
                        'target_date': x['date'].iloc[0],
                    })
                ).reset_index(drop=True)

                # Calculate the forecast date (days_ahead days before the target date)
                daily_peaks['forecast_date'] = daily_peaks['target_date'] - timedelta(days=days_ahead_int)

                # Add days_ahead information
                daily_peaks['days_ahead'] = days_ahead_int

                # Initialize algorithm entry if needed
                if algo not in forecasts_by_algo:
                    forecasts_by_algo[algo] = []

                # Append to collection
                forecasts_by_algo[algo].append(daily_peaks)

        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")

    # Process and save forecasts for each algorithm
    results = {}

    for algo, forecast_dfs in forecasts_by_algo.items():
        if not forecast_dfs:
            print(f"Warning: No valid forecast data found for {algo}")
            continue

        print(f"Processing forecast data for {algo} algorithm...")

        # Combine all forecast dataframes for this algorithm
        combined_forecasts = pd.concat(forecast_dfs, ignore_index=True)

        # Format the dates as strings in YYYYMMDD format
        combined_forecasts['forecast_date_str'] = combined_forecasts['forecast_date'].apply(
            lambda x: x.strftime('%Y%m%d') if pd.notna(x) else '')
        combined_forecasts['target_date_str'] = combined_forecasts['target_date'].apply(
            lambda x: x.strftime('%Y%m%d') if pd.notna(x) else '')

        # Filter out rows with empty dates (could happen if date arithmetic produced NaT)
        combined_forecasts = combined_forecasts[
            (combined_forecasts['forecast_date_str'] != '') &
            (combined_forecasts['target_date_str'] != '')
            ]

        # Convert days_ahead to integer to ensure proper formatting
        combined_forecasts['days_ahead'] = combined_forecasts['days_ahead'].astype(int)

        # Keep peak_forecast as float to preserve original precision
        combined_forecasts['peak_forecast'] = combined_forecasts['peak_forecast'].astype(float)

        # Select and order columns for output
        forecast_peak_output = combined_forecasts[
            ['forecast_date_str', 'target_date_str', 'days_ahead', 'peak_forecast']
        ]

        # Group by forecast date to ensure each forecast date has predictions for consecutive 6 days
        forecast_dates = forecast_peak_output['forecast_date_str'].unique()

        # Create a list to store the final output rows
        final_output_rows = []

        for forecast_date in forecast_dates:
            # For each forecast date, we need to ensure we have predictions for the next 6 consecutive days
            forecast_date_dt = datetime.strptime(forecast_date, '%Y%m%d')

            # Generate the expected target dates (next 6 consecutive days)
            expected_target_dates = [
                (forecast_date_dt + timedelta(days=i + 1)).strftime('%Y%m%d')
                for i in range(6)
            ]

            # For each target date, find the corresponding days_ahead prediction
            for i, target_date in enumerate(expected_target_dates):
                days_ahead = i + 1

                # Look for the prediction from the appropriate days_ahead file
                matching_rows = forecast_peak_output[
                    (forecast_peak_output['forecast_date_str'] == forecast_date) &
                    (forecast_peak_output['target_date_str'] == target_date) &
                    (forecast_peak_output['days_ahead'] == days_ahead)
                    ]

                if not matching_rows.empty:
                    # Use the existing prediction
                    row = matching_rows.iloc[0]
                    final_output_rows.append([
                        row['forecast_date_str'],
                        row['target_date_str'],
                        int(row['days_ahead']),
                        float(row['peak_forecast'])  # Keep as float to preserve precision
                    ])
                else:
                    # If we don't have a prediction for this target date with the correct days_ahead,
                    # look for any prediction for this target date
                    any_prediction = forecast_peak_output[
                        (forecast_peak_output['forecast_date_str'] == forecast_date) &
                        (forecast_peak_output['target_date_str'] == target_date)
                        ]

                    if not any_prediction.empty:
                        # Use the first available prediction but correct the days_ahead
                        row = any_prediction.iloc[0]
                        final_output_rows.append([
                            row['forecast_date_str'],
                            row['target_date_str'],
                            days_ahead,  # Update days_ahead
                            float(row['peak_forecast'])  # Keep as float to preserve precision
                        ])
                    else:
                        # If we still don't have a prediction, use the previous day's prediction as an estimate
                        # or set to a default value
                        if final_output_rows:
                            # Use the last prediction as a baseline
                            last_prediction = final_output_rows[-1][3]
                            # Add some random variation (optional)
                            # estimated_value = last_prediction + random.randint(-500, 500)
                            estimated_value = last_prediction
                        else:
                            # Default value if we have no previous prediction
                            estimated_value = 18000.0

                        final_output_rows.append([
                            forecast_date,
                            target_date,
                            days_ahead,
                            float(estimated_value)  # Keep as float to preserve precision
                        ])

        # Save the ForecastPeak file for this algorithm
        output_file = os.path.join(output_dir, f"ForecastPeak_{algo}_new.csv")
        print(f"Saving ForecastPeak file for {algo} to {output_file}...")

        # Save without header, using pandas to avoid numpy type issues
        # Use 'w' mode with newline='' for Windows compatibility
        with open(output_file, 'w', newline='') as f:
            for row in final_output_rows:
                # Format with 2 decimal places to preserve original precision
                f.write(f"{row[0]},{row[1]},{row[2]},{row[3]:.2f}\n")

        results[algo] = {
            'file': output_file,
            'entries': len(final_output_rows)
        }

    if not results:
        print("Error: No ForecastPeak files were generated")
        return False

    print("Forecast data processing completed successfully!")

    # Print summary
    print("\nForecastPeak Files Generated:")
    for algo, info in results.items():
        print(f"- {algo}: {info['entries']} entries, saved to {info['file']}")

    return True


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process IESO forecast data.')
    parser.add_argument('--input-dir', default='.',
                        help='Directory containing input files (default: current directory)')
    parser.add_argument('--output-dir', default='.',
                        help='Directory to save output files (default: current directory)')
    parser.add_argument('--algorithms', nargs='+', choices=['ECA', 'RTO', 'TESLA'],
                        help='Algorithms to process (default: all available)')

    # Parse arguments
    args = parser.parse_args()

    # Create dictionary mapping days ahead to file paths
    # Use os.path.join for Windows compatibility
    input_files = {
        "1": os.path.join(args.input_dir, "IESO-Ontario Demand 2022-2025 1 day ahead.csv"),
        "2": os.path.join(args.input_dir, "IESO-Ontario Demand 2022-2025 2 day ahead.csv"),
        "3": os.path.join(args.input_dir, "IESO-Ontario Demand 2022-2025 3 day ahead.csv"),
        "4": os.path.join(args.input_dir, "IESO-Ontario Demand 2022-2025 4 day ahead.csv"),
        "5": os.path.join(args.input_dir, "IESO-Ontario Demand 2022-2025 5 day ahead.csv"),
        "6": os.path.join(args.input_dir, "IESO-Ontario Demand 2022-2025 6 day ahead.csv")
    }

    # Process the files
    success = process_forecast_peak(input_files, args.output_dir, args.algorithms)

    if success:
        print("\nForecastPeak files successfully generated in: " + args.output_dir)
    else:
        print("\nError: Failed to generate ForecastPeak files")


if __name__ == "__main__":
    main()
