import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import argparse

def process_actual_demand(input_files, output_file):
    """
    Process actual demand data from multiple files to create ActualDemand file.
    
    Args:
        input_files: Dictionary mapping day-ahead values to file paths
        output_file: Path to save the ActualDemand file
    """
    print("Starting actual demand data processing...")
    
    # Initialize list to store all actual demand data
    all_actuals = []
    
    # Process each file to extract actual demand
    for days_ahead, file_path in input_files.items():
        print(f"Extracting actual demand from {days_ahead} day ahead file: {file_path}")
        
        try:
            # Try different date formats since files might have different formats
            try:
                df = pd.read_csv(file_path, parse_dates=['Date'])
            except:
                df = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=False)
                
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
            
            # Use TESLA actual demand column for actual values
            actual_col = [col for col in df.columns if 'Actual' in col]
            if not actual_col:
                print(f"Warning: No actual demand column found in {file_path}")
                continue
                
            actual_col = actual_col[0]
            
            # Create a dataframe for actuals
            actual_df = df[['Datetime', actual_col]].copy()
            actual_df.rename(columns={actual_col: 'actual_value'}, inplace=True)
            actual_df['date'] = actual_df['Datetime'].dt.date
            actual_df['hour'] = actual_df['Datetime'].dt.hour
            
            # Append to our collection
            all_actuals.append(actual_df)
            
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
    
    if not all_actuals:
        print("Error: No valid actual demand data found in input files")
        return False
    
    # Combine all actual dataframes and remove duplicates
    combined_actuals = pd.concat(all_actuals, ignore_index=True)
    combined_actuals = combined_actuals.drop_duplicates(subset=['Datetime'])
    
    # Filter out rows with NaN actual values
    combined_actuals = combined_actuals.dropna(subset=['actual_value'])
    
    # Initialize empty list to store daily peaks
    daily_peak_list = []
    
    # Process each date group safely
    for date, group in combined_actuals.groupby('date'):
        if group.empty or group['actual_value'].isna().all():
            print(f"Warning: No valid actual demand data for date {date}, skipping...")
            continue
            
        # Find the peak demand and corresponding hour
        try:
            peak_value = group['actual_value'].max()
            # Only proceed if we have a valid peak
            if pd.notna(peak_value):
                peak_idx = group['actual_value'].idxmax()
                peak_hour = group.loc[peak_idx, 'hour']
                
                daily_peak_list.append({
                    'date': date,
                    'peak_demand': peak_value,
                    'peak_hour': peak_hour
                })
            else:
                print(f"Warning: No valid peak found for date {date}, skipping...")
        except Exception as e:
            print(f"Error processing peak for date {date}: {str(e)}, skipping...")
    
    # Convert list to DataFrame
    if not daily_peak_list:
        print("Error: No valid daily peaks found in the data")
        return False
        
    daily_peaks = pd.DataFrame(daily_peak_list)
    
    # Format the ActualDemand file
    daily_peaks['date_str'] = daily_peaks['date'].apply(lambda x: x.strftime('%Y%m%d'))
    daily_peaks['peak_hour_str'] = daily_peaks['peak_hour'].apply(lambda x: f"{int(x):02d}")
    daily_peaks['date_hour'] = daily_peaks['date_str'] + daily_peaks['peak_hour_str']
    
    actual_demand_output = daily_peaks[['date_str', 'peak_demand', 'peak_hour', 'date_hour']]
    
    # Save the ActualDemand file
    print(f"Saving ActualDemand file to {output_file}...")
    actual_demand_output.to_csv(output_file, index=False, header=False)
    
    print(f"ActualDemand file generated with {len(actual_demand_output)} days of peak demand data")
    print(f"Date range covered: {min(daily_peaks['date'])} to {max(daily_peaks['date'])}")
    
    return True

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process IESO actual demand data.')
    parser.add_argument('--input-dir', default='/home/ubuntu/upload', 
                        help='Directory containing input files')
    parser.add_argument('--output-file', default='/home/ubuntu/ActualDemand_new.csv', 
                        help='Path to save the ActualDemand file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create dictionary mapping days ahead to file paths
    input_files = {
        "1": os.path.join(args.input_dir, "IESO-Ontario Demand 2022-2025 1 day ahead.csv"),
        "2": os.path.join(args.input_dir, "IESO-Ontario Demand 2022-2025 2 day ahead.csv"),
        "3": os.path.join(args.input_dir, "IESO-Ontario Demand 2022-2025 3 day ahead.csv"),
        "4": os.path.join(args.input_dir, "IESO-Ontario Demand 2022-2025 4 day ahead.csv"),
        "5": os.path.join(args.input_dir, "IESO-Ontario Demand 2022-2025 5 day ahead.csv"),
        "6": os.path.join(args.input_dir, "IESO-Ontario Demand 2022-2025 6 day ahead.csv")
    }
    
    # Process the files
    success = process_actual_demand(input_files, args.output_file)
    
    if success:
        print(f"\nActualDemand file successfully generated: {args.output_file}")
    else:
        print("\nError: Failed to generate ActualDemand file")

if __name__ == "__main__":
    main()
