import os
import glob
import json
import pandas as pd
import argparse

def main(result_dirs):
    # Find all results.json files
    results = sorted(glob.glob(os.path.join(result_dirs, '*', 'train', 'ours_30000', 'results.json')))
    elapseds = sorted(glob.glob(os.path.join(result_dirs, '*', 'training_time.json')))
    
    # Initialize lists to store the data
    data = []
    
    # Read each JSON file and extract the metrics
    for result_file, elapsed_file in zip(results, elapseds):
        with open(result_file, 'r') as f:
            result = json.load(f)
        with open(elapsed_file, 'r') as f:
            elapsed = json.load(f)
        
        # Extract the experiment name from the file path
        exp_name = result_file.split('/')[-4]
        
        # Extract the metrics
        cd = result['overall']
        
        # Append the data to the list
        data.append({
            'Experiment': exp_name,
            'CD': cd,
            'Time': (elapsed['stop_time'] - elapsed['start_time']) / 60.
        })
    
    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    # Sort the DataFrame by experiment name
    df = df.sort_values('Experiment')

    # Display the table
    print(df.to_string(index=False))

    # calculate average PSNR, SSIM, and LPIPS
    avg_cd = df['CD'].mean()
    avg_time = df['Time'].mean()

    print(f"Average CD: {avg_cd}")
    print(f"Average Time: {avg_time} min.")
    # Optionally, save the table to a CSV file
    # df.to_csv('results_summary.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process results from JSON files.")
    parser.add_argument("--model_path", "-m", help="model path")
    args = parser.parse_args()
    main(args.model_path)