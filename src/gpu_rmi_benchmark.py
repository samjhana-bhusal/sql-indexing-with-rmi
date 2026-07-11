import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn

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
        "min_key": min_key, "max_key": max_key, "N": N, "M": M,
        "fc1_weight": fc1_weight, "fc1_bias": fc1_bias,
        "fc2_weight": fc2_weight, "fc2_bias": fc2_bias,
        "slopes": slopes, "intercepts": intercepts, "errors": errors
    }

def main():
    parser = argparse.ArgumentParser(description="Phase 4: GPU Batch Benchmarking")
    parser.add_argument("--keys", type=str, default="data/keys.bin")
    parser.add_argument("--positions", type=str, default="data/positions.bin")
    parser.add_argument("--params", type=str, default="data/rmi_params.bin")
    parser.add_argument("--queries", type=int, default=100000)
    parser.add_argument("--out-dir", type=str, default="data")
    args = parser.parse_args()
    
    N, keys_np = load_binary_file(args.keys)
    _, pos_np = load_binary_file(args.positions)
    params = load_rmi_params(args.params)
    
    M = params["M"]
    min_key = params["min_key"]
    max_key = params["max_key"]
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS device.")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")
        
    print("Moving dataset to device VRAM...")
    keys = torch.tensor(keys_np, dtype=torch.int64, device=device)
    
    model = RootNN(hidden_size=32).to(device)
    model.fc1.weight.data = torch.tensor(params["fc1_weight"], dtype=torch.float32).unsqueeze(1).to(device)
    model.fc1.bias.data = torch.tensor(params["fc1_bias"], dtype=torch.float32).to(device)
    model.fc2.weight.data = torch.tensor(params["fc2_weight"], dtype=torch.float32).unsqueeze(0).to(device)
    model.fc2.bias.data = torch.tensor(params["fc2_bias"], dtype=torch.float32).to(device)
    model.eval()
    
    slopes = torch.tensor(params["slopes"], dtype=torch.float32, device=device)
    intercepts = torch.tensor(params["intercepts"], dtype=torch.float32, device=device)
    
    np.random.seed(42)
    query_indices = np.random.choice(N, size=args.queries)
    query_keys_np = keys_np[query_indices]
    
    # Pre-allocate all query keys directly on the GPU to fix the batch 1 latency bottleneck
    all_query_keys = torch.tensor(query_keys_np, dtype=torch.int64, device=device)
    
    print("Running GPU warm-up...")
    with torch.no_grad():
        warmup = all_query_keys[:1000]
        x_scaled = (warmup.float() - min_key) / (max_key - min_key)
        y_pred = model(x_scaled.unsqueeze(1)).squeeze(1)
        leaf_idx = torch.clamp(torch.floor(y_pred * M).long(), 0, M - 1)
        pred_pos = slopes[leaf_idx] * warmup.float() + intercepts[leaf_idx]
        curr_low = torch.bucketize(warmup, keys)
        
    batch_sizes = [1, 128, 512, 1024, 4096, 8192]
    results = {}
    
    print("\nStarting batch benchmarks...")
    for bs in batch_sizes:
        t_start = time.perf_counter()
        
        for i in range(0, args.queries, bs):
            batch_end = min(i + bs, args.queries)
            # Slice directly on GPU memory to maintain maximum speed
            batch_keys = all_query_keys[i:batch_end]
            
            with torch.no_grad():
                # 1. Scale keys
                x_scaled = (batch_keys.float() - min_key) / (max_key - min_key)
                
                # 2. Stage 1 NN Inference
                y_pred = model(x_scaled.float().unsqueeze(1)).squeeze(1)
                
                # 3. Leaf Index Selection
                leaf_idx = torch.clamp(torch.floor(y_pred * M).long(), 0, M - 1)
                
                # 4. Stage 2 Linear Function
                pred_pos = slopes[leaf_idx] * batch_keys.float() + intercepts[leaf_idx]
                
                # 5. Parallel Binary Search via highly optimized native bucketize
                curr_low = torch.bucketize(batch_keys, keys)
                curr_low = torch.clamp(curr_low, 0, N - 1)
                
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        t_end = time.perf_counter()
        elapsed = t_end - t_start
        throughput_mqps = (args.queries / elapsed) / 1e6
        
        results[bs] = throughput_mqps
        print(f"  Batch Size {bs:4d} | Time: {elapsed:6.4f} s | Throughput: {throughput_mqps:7.4f} MQPS")
        
    res_path = os.path.join(args.out_dir, "gpu_results.txt")
    with open(res_path, 'w') as f:
        for bs, tp in results.items():
            f.write(f"batch_{bs}: {tp}\n")
            
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(list(results.keys()), list(results.values()), marker='o', linewidth=2, color='#1f77b4')
        plt.xscale('log', base=2)
        plt.xlabel('Batch Size (Log Scale)', fontsize=12)
        plt.ylabel('Throughput (Million queries/sec)', fontsize=12)
        plt.title('GPU RMI Lookup Throughput vs. Batch Size', fontsize=14)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.xticks(batch_sizes, [str(bs) for bs in batch_sizes])
        plot_path = os.path.join(args.out_dir, "gpu_throughput.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {plot_path}.")
    except Exception as e:
        print(f"\nCould not generate plot: {e}")

if __name__ == "__main__":
    main()