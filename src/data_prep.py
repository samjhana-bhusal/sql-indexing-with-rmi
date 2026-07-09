import os
import sys
import argparse
import urllib.request
import gzip
import numpy as np

def generate_synthetic_data(num_elements=10000000, distribution="lognormal"):
    print(f"Generating {num_elements} synthetic uint64 keys using '{distribution}' distribution...")
    np.random.seed(42)
    
    if distribution == "uniform":
        # Uniformly spaced keys with tiny random noise
        base = np.linspace(1000000000000, 5000000000000, num_elements, dtype=np.float64)
        noise = np.random.uniform(-100, 100, size=num_elements)
        keys = (base + noise).astype(np.uint64)
        
    elif distribution == "normal":
        # Normally distributed keys
        samples = np.random.normal(loc=3000000000000, scale=500000000000, size=num_elements)
        # Shift to positive range and convert to uint64
        keys = samples.astype(np.uint64)
        
    elif distribution == "lognormal":
        # Cumulative sum of log-normal steps (skewed intervals)
        steps = np.random.lognormal(mean=2.0, sigma=1.0, size=num_elements)
        keys = np.cumsum(steps).astype(np.uint64)
        base_offset = np.uint64(1000000000000)
        keys += base_offset
        
    elif distribution == "clustered":
        # Multiple dense clusters with large empty gaps (hard CDF to learn)
        num_clusters = 5
        keys_per_cluster = num_elements // num_clusters
        keys_list = []
        
        # Generate clusters centered at different intervals
        centroids = [1e12, 1.8e12, 3.2e12, 4.0e12, 4.8e12]
        for c in centroids:
            samples = np.random.normal(loc=c, scale=1e9, size=keys_per_cluster)
            keys_list.append(samples.astype(np.uint64))
            
        keys = np.concatenate(keys_list)
        
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
        
    # Deduplicate and sort
    keys = np.unique(keys)
    print(f"Generated {len(keys)} unique keys after deduplication.")
    return keys

def download_sosd_dataset(dest_dir="data"):
    url = "https://zenodo.org/records/7841164/files/wiki_ts_200M_uint64.gz"
    gz_path = os.path.join(dest_dir, "wiki_ts_200M_uint64.gz")
    bin_path = os.path.join(dest_dir, "wiki_ts_200M_uint64")
    
    if os.path.exists(bin_path):
        print(f"SOSD binary dataset already exists at {bin_path}.")
        return bin_path
        
    print(f"Attempting to download wiki_ts_200M_uint64 from Zenodo: {url}...")
    try:
        os.makedirs(dest_dir, exist_ok=True)
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\rDownloading: {percent}% completed")
            sys.stdout.flush()
            
        urllib.request.urlretrieve(url, gz_path, progress_hook)
        print("\nDownload complete. Decompressing...")
        
        with gzip.open(gz_path, 'rb') as f_in:
            with open(bin_path, 'wb') as f_out:
                f_out.write(f_in.read())
        
        print(f"Decompressed to {bin_path}.")
        os.remove(gz_path)
        return bin_path
    except Exception as e:
        print(f"\nFailed to download SOSD dataset: {e}")
        print("Falling back to synthetic data generation.")
        return None

def load_sosd_binary(bin_path):
    print(f"Loading SOSD binary dataset from {bin_path}...")
    with open(bin_path, 'rb') as f:
        n_bytes = f.read(8)
        N = np.frombuffer(n_bytes, dtype=np.uint64)[0]
        data = np.fromfile(f, dtype=np.uint64)
        return data

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Data Acquisition & Preprocessing")
    parser.add_argument("--download", action="store_true", help="Try to download real SOSD wiki_ts dataset")
    parser.add_argument("--distribution", type=str, default="lognormal", choices=["uniform", "normal", "lognormal", "clustered"], help="Distribution type for synthetic data")
    parser.add_argument("--num-keys", type=int, default=10000000, help="Number of keys to generate if synthetic")
    parser.add_argument("--out-dir", type=str, default="data", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    keys = None
    if args.download:
        bin_path = download_sosd_dataset(args.out_dir)
        if bin_path and os.path.exists(bin_path):
            keys = load_sosd_binary(bin_path)
            
    if keys is None:
        keys = generate_synthetic_data(args.num_keys, args.distribution)
        
    # Sort and Deduplicate
    print("Sorting and deduplicating...")
    keys = np.sort(keys)
    keys = np.unique(keys)
    
    # Assertions check
    print("Running assertion checks...")
    assert len(keys) > 0, "Dataset cannot be empty"
    assert np.all(np.diff(keys) >= 0), "Keys must be strictly sorted in ascending order"
    print(f"Assertion passed: all {len(keys)} keys are sorted.")
    
    # Map to positions
    N = len(keys)
    positions = np.arange(N, dtype=np.uint64)
    
    # Write preprocessed datasets
    keys_out = os.path.join(args.out_dir, "keys.bin")
    pos_out = os.path.join(args.out_dir, "positions.bin")
    
    print(f"Writing keys to {keys_out}...")
    with open(keys_out, 'wb') as f:
        f.write(np.uint64(N).tobytes())
        f.write(keys.tobytes())
        
    print(f"Writing positions to {pos_out}...")
    with open(pos_out, 'wb') as f:
        f.write(np.uint64(N).tobytes())
        f.write(positions.tobytes())
        
    # Write metadata
    meta_out = os.path.join(args.out_dir, "metadata.txt")
    print(f"Writing metadata to {meta_out}...")
    with open(meta_out, 'w') as f:
        f.write(f"N: {N}\n")
        f.write(f"min_key: {keys[0]}\n")
        f.write(f"max_key: {keys[-1]}\n")
        f.write(f"distribution: {args.distribution}\n")
        
    print("Phase 1 complete! Data is ready.")

if __name__ == "__main__":
    main()
