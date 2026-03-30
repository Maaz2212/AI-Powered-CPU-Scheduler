import pandas as pd
from pathlib import Path

base_dir = Path(__file__).resolve().parent
old_csv = base_dir / "process_data.csv"
new_csv = base_dir / "process_data_collected.csv"

def main():
    if not old_csv.exists() or not new_csv.exists():
        print(f"Error: Could not find one or both CSV files in {base_dir}")
        return

    print("Loading datasets...")
    df_old = pd.read_csv(old_csv)
    df_new = pd.read_csv(new_csv)
    
    print(f"Original dataset rows: {len(df_old)}")
    print(f"Collected dataset rows: {len(df_new)}")

    cols_to_drop = ["burst_time", "memory_rss"]
    existing_drops = [c for c in cols_to_drop if c in df_new.columns]
    
    if existing_drops:
        print(f"Dropping columns from collected dataset: {existing_drops}")
        df_new = df_new.drop(columns=existing_drops)

    # Concatenate them
    print("Concatenating...")
    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    
    print(f"Combined dataset rows: {len(df_combined)}")
    df_combined.to_csv(old_csv, index=False)
    print(f"Successfully saved to {old_csv.name}")

if __name__ == "__main__":
    main()
