import time
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from inference import SatelliteImageMatcher


def resolve_path(path_str: str, metadata_dir: Path) -> Path:
    """
    Helper function to resolve tile paths whether the script is run from root or task2 folder.
    """
    p = Path(path_str)
    if p.exists():
        return p
    
    # Try resolving relative to metadata processed_pairs directory
    alt_p = metadata_dir / p.name
    if alt_p.exists():
        return alt_p
        
    alt_p2 = metadata_dir.parent / p
    if alt_p2.exists():
        return alt_p2
        
    return p

def run_benchmark(metadata_path: str = "task2/data/pairs_metadata.csv", output_csv: str = "task2/data/benchmark_results.csv", sample_size: int = 50):
    """
    Automatically assesses the quality of satellite image matching across the entire dataset
    """
    metadata_file = Path(metadata_path)

    # Fallback if executed directly inside the task2 folder
    if not metadata_file.exists():
        metadata_file = Path("data/pairs_metadata.csv")
        output_csv = "data/benchmark_results.csv"

    if not metadata_file.exists():
        print(f"Error: Metadata file '{metadata_path}' not found.")
        return

    metadata_dir = metadata_file.parent / "processed_pairs"

    # Loading the pair metadata
    df_pairs = pd.read_csv(metadata_file)
    
    if sample_size and len(df_pairs) > sample_size:        
        df_pairs = df_pairs.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # Initialize the model
    matcher = SatelliteImageMatcher(confidence_threshold=0.2)

    results = []

    for idx, row in tqdm(df_pairs.iterrows(), total=len(df_pairs), desc="Evaluating pairs"):
        tile_pair_id = row.get('tile_pair_id', f"pair_{idx}")
        
        # Resolve image paths
        img1_path = resolve_path(row['tile1_path'], metadata_dir)
        img2_path = resolve_path(row['tile2_path'], metadata_dir)

        if not img1_path.exists() or not img2_path.exists():
            print(f"\nSkipping {tile_pair_id}: Files not found ({img1_path})")
            continue

        try:
            # Measure execution latency
            start_time = time.time()
            img1, img2, kpts1, kpts2, inliers_mask = matcher.match_pair(str(img1_path), str(img2_path))
            latency_ms = (time.time() - start_time) * 1000.0

            total_matches = len(kpts1)
            inliers_count = int(np.sum(inliers_mask))
            outliers_count = total_matches - inliers_count
            inlier_ratio = (inliers_count / total_matches) if total_matches > 0 else 0.0

            results.append({
                'tile_pair_id': tile_pair_id,
                'total_matches': total_matches,
                'inliers_count': inliers_count,
                'outliers_count': outliers_count,
                'inlier_ratio': round(inlier_ratio, 4),
                'latency_ms': round(latency_ms, 2)
            })

        except Exception as e:
            print(f"\nFailed processing pair {tile_pair_id}: {e}")
            continue

    # Form a summary dataframe
    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("\nError: No pairs were processed successfully. Check tile paths in processed_pairs folder.")
        return

    results_df.to_csv(output_csv, index=False)

    print("\nBENCHMARK SUMMARY REPORT")
    print(f"Total Pairs Processed : {len(results_df)}")
    print(f"Avg Latency per Pair  : {results_df['latency_ms'].mean():.2f} ms")
    print(f"Mean Total Matches    : {results_df['total_matches'].mean():.1f}")
    print(f"Mean Inliers Count    : {results_df['inliers_count'].mean():.1f}")
    print(f"Mean Inlier Ratio     : {results_df['inlier_ratio'].mean() * 100:.2f}%")


if __name__ == "__main__":
    run_benchmark()