import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn

# Stage 1 NN model structure
class RootNN(nn.Module):
    def __init__(self, hidden_size=32):
        super(RootNN, self).__init__()
        self.fc1 = nn.Linear(1, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def load_binary_file(path):
    with open(path, 'rb') as f:
        N = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        data = np.fromfile(f, dtype=np.uint64)
        return N, data

def load_rmi_params(path):
    with open(path, 'rb') as f:
        min_key = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        max_key = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        N = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        M = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        
        fc1_weight = np.frombuffer(f.read(32 * 8), dtype=np.float64)
        fc1_bias = np.frombuffer(f.read(32 * 8), dtype=np.float64)
        fc2_weight = np.frombuffer(f.read(32 * 8), dtype=np.float64)
        fc2_bias = np.frombuffer(f.read(1 * 8), dtype=np.float64)
        
        slopes = np.frombuffer(f.read(int(M) * 8), dtype=np.float64)
        intercepts = np.frombuffer(f.read(int(M) * 8), dtype=np.float64)
        errors = np.frombuffer(f.read(int(M) * 8), dtype=np.uint64)
        
    return {
        "min_key": min_key,
        "max_key": max_key,
        "N": N,
        "M": M,
        "fc1_weight": fc1_weight,
        "fc1_bias": fc1_bias,
        "fc2_weight": fc2_weight,
        "fc2_bias": fc2_bias,
        "slopes": slopes,
        "intercepts": intercepts,
        "errors": errors
    }

def main():
    parser = argparse.ArgumentParser(description="Phase 4: GPU Batch Benchmarking")
    parser.add_argument("--keys", type=str, default="data/keys.bin", help="Path to keys.bin")
    parser.add_argument("--positions", type=str, default="data/positions.bin", help="Path to positions.bin")
    parser.add_argument("--params", type=str, default="data/rmi_params.bin", help="Path to rmi_params.bin")
    parser.add_argument("--queries", type=int, default=100000, help="Number of random queries to run")
    parser.add_argument("--out-dir", type=str, default="data", help="Output directory")
    args = parser.parse_args()
    
    # Load dataset
    N, keys_np = load_binary_file(args.keys)
    _, pos_np = load_binary_file(args.positions)
    
    # Load parameters
    params = load_rmi_params(args.params)
    M = params["M"]
    min_key = params["min_key"]
    max_key = params["max_key"]
    
    # Device selection (CUDA or Apple Silicon MPS or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS device.")
    else:
        device = torch.device("cpu")
        print("Using CPU device (no GPU detected).")
        
    # Copy arrays to tensor
    print("Moving dataset to device VRAM...")
    keys = torch.tensor(keys_np, dtype=torch.int64, device=device) # using int64 in PyTorch since uint64 is not fully supported on MPS/CUDA
    positions = torch.tensor(pos_np, dtype=torch.int64, device=device)
    
    # Define Stage 1 NN on GPU and load weights
    model = RootNN(hidden_size=32).to(device)
    
    # Set weights from trained parameters
    # Reshape weights appropriately: fc1 weight (32, 1), fc2 weight (1, 32)
    model.fc1.weight.data = torch.tensor(params["fc1_weight"], dtype=torch.float32).unsqueeze(1).to(device)
    model.fc1.bias.data = torch.tensor(params["fc1_bias"], dtype=torch.float32).to(device)
    model.fc2.weight.data = torch.tensor(params["fc2_weight"], dtype=torch.float32).unsqueeze(0).to(device)
    model.fc2.bias.data = torch.tensor(params["fc2_bias"], dtype=torch.float32).to(device)
    model.eval()
    
    # Stage 2 parameters on GPU
    slopes = torch.tensor(params["slopes"], dtype=torch.float64, device=device)
    intercepts = torch.tensor(params["intercepts"], dtype=torch.float64, device=device)
    errors = torch.tensor(params["errors"], dtype=torch.int64, device=device)
    
    # Select random queries
    np.random.seed(42)
    query_indices = np.random.choice(N, size=args.queries)
    query_keys_np = keys_np[query_indices]
    query_positions_np = pos_np[query_indices]
    
    # Warm-up to trigger JIT and cache allocations
    print("Running GPU warm-up...")
    warmup_queries = torch.tensor(query_keys_np[:1000], dtype=torch.int64, device=device)
    with torch.no_grad():
        # Stage 1
        x_scaled = (warmup_queries.double() - min_key) / (max_key - min_key)
        y_pred = model(x_scaled.float().unsqueeze(1)).squeeze(1)
        leaf_idx = torch.clamp(torch.floor(y_pred * M).long(), 0, M - 1)
        # Stage 2
        pred_pos = slopes[leaf_idx] * warmup_queries.double() + intercepts[leaf_idx]
        eps = errors[leaf_idx]
        low = torch.clamp((pred_pos.long() - eps), 0, N - 1)
        high = torch.clamp((pred_pos.long() + eps), 0, N - 1)
        
    # Synchronize if CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    batch_sizes = [1, 128, 512, 1024, 4096, 8192]
    results = {}
    
    print("\nStarting batch benchmarks...")
    for bs in batch_sizes:
        t_start = time.perf_counter()
        
        # Accumulate predictions in batches
        all_found_positions = []
        
        for i in range(0, args.queries, bs):
            batch_end = min(i + bs, args.queries)
            batch_keys = torch.tensor(query_keys_np[i:batch_end], dtype=torch.int64, device=device)
            
            with torch.no_grad():
                # 1. Scale keys for Stage 1 Neural Network
                x_scaled = (batch_keys.double() - min_key) / (max_key - min_key)
                
                # 2. Stage 1 Inference
                # Input shape (B, 1), output shape (B,)
                y_pred = model(x_scaled.float().unsqueeze(1)).squeeze(1)
                
                # 3. Select Leaf Model index and clamp
                leaf_idx = torch.clamp(torch.floor(y_pred * M).long(), 0, M - 1)
                
                # 4. Stage 2 Linear Spline Inference
                pred_pos = slopes[leaf_idx] * batch_keys.double() + intercepts[leaf_idx]
                eps = errors[leaf_idx]
                
                # 5. Define Search bounds
                low = torch.clamp((pred_pos.long() - eps), 0, N - 1)
                high = torch.clamp((pred_pos.long() + eps), 0, N - 1)
                
                # 6. Parallel Bounded Binary Search in VRAM
                # Maximum iterations needed based on max error in this batch
                max_window = torch.max(high - low).item()
                num_iters = int(np.ceil(np.log2(max_window + 1))) if max_window > 0 else 1
                
                curr_low = low.clone()
                curr_high = high.clone()
                
                for _ in range(num_iters):
                    mid = (curr_low + curr_high) // 2
                    val = keys[mid]
                    go_right = (val < batch_keys)
                    curr_low = torch.where(go_right, mid + 1, curr_low)
                    curr_high = torch.where(go_right, curr_high, mid)
                
                # Verification on GPU (assert correctness)
                found_keys = keys[curr_low]
                correct = (found_keys == batch_keys)
                assert torch.all(correct), "Parallel Binary Search returned incorrect keys!"
                
        # Synchronize
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            # MPS doesn't have synchronize, but standard operations are synchronous on host thread boundaries
            pass
            
        t_end = time.perf_counter()
        elapsed = t_end - t_start
        throughput = args.queries / elapsed
        throughput_mqps = throughput / 1e6
        
        results[bs] = throughput_mqps
        print(f"  Batch Size {bs:4d} | Time: {elapsed:6.4f} s | Throughput: {throughput_mqps:7.4f} MQPS")
        
    # Export results to text file
    res_path = os.path.join(args.out_dir, "gpu_results.txt")
    with open(res_path, 'w') as f:
        for bs, tp in results.items():
            f.write(f"batch_{bs}: {tp}\n")
            
    # Try plotting
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(list(results.keys()), list(results.values()), marker='o', linewidth=2, color='#1f77b4')
        plt.xscale('log', base=2)
        plt.xlabel('Batch Size (Log Scale)', fontsize=12)
        plt.ylabel('Throughput (Million queries/sec)', fontsize=12)
        plt.title('GPU RMI Lookup Throughput vs. Batch Size', fontsize=14)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        
        # Style layout
        plt.xticks(batch_sizes, [str(bs) for bs in batch_sizes])
        plot_path = os.path.join(args.out_dir, "gpu_throughput.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {plot_path}.")
    except Exception as e:
        print(f"\nCould not generate plot: {e} (Results written to {res_path})")

if __name__ == "__main__":
    main()
