import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class RootNN(nn.Module):
    def __init__(self, hidden_size=32):
        super(RootNN, self).__init__()
        self.fc1 = nn.Linear(1, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def load_binary_file(path):
    with open(path, 'rb') as f:
        N = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        data = np.fromfile(f, dtype=np.uint64)
        return data

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Train RMI Hybrid Learned Index with Adaptive Epochs")
    parser.add_argument("--keys", type=str, default="data/keys.bin", help="Path to keys.bin")
    parser.add_argument("--positions", type=str, default="data/positions.bin", help="Path to positions.bin")
    parser.add_argument("--num-leaves", type=int, default=1000, help="Number of leaf models (M)")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size for Stage 1 training")
    parser.add_argument("--out-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--run-id", type=str, default="", help="Run ID directory mapping")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs") # backward compatibility; it is dynamically handled

    args = parser.parse_args()
    
    out_dir   = args.out_dir
    keys_path = args.keys
    pos_path  = args.positions
    
    # run-id: only auto-derive paths when keys/positions are still at their defaults.
    # If the caller (e.g. parameter_sweep.py) passed explicit --keys / --positions,
    # respect those and just update out_dir.
    keys_explicitly_set = (args.keys != "data/keys.bin")
    if args.run_id:
        out_dir = os.path.join(args.out_dir, args.run_id)
        if not keys_explicitly_set:
            keys_path = os.path.join(out_dir, "keys.bin")
            pos_path  = os.path.join(out_dir, "positions.bin")
        
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading keys from   : {keys_path}")
    print(f"Loading positions from: {pos_path}")
    print(f"Output directory    : {out_dir}")
    keys = load_binary_file(keys_path)
    positions = load_binary_file(pos_path)
    N = len(keys)
    M = args.num_leaves
    
    min_key = keys[0]
    max_key = keys[-1]
    
    print(f"Dataset has {N} elements. Key range: [{min_key}, {max_key}]")
    print(f"Number of leaf models (M): {M}")
    
    scaled_keys = (keys.astype(np.float64) - min_key) / (max_key - min_key)
    scaled_positions = positions.astype(np.float64) / N
    
    sample_size = min(1000000, N)
    sample_indices = np.random.choice(N, size=sample_size, replace=False)
    
    X_train = torch.tensor(scaled_keys[sample_indices], dtype=torch.float32).unsqueeze(1)
    Y_train = torch.tensor(scaled_positions[sample_indices], dtype=torch.float32).unsqueeze(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = RootNN(hidden_size=32).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Dynamic Training Loop Parameters
    max_epochs = 40          
    min_epochs = 5           
    loss_threshold = 1e-6    
    
    model.train()
    prev_loss = float('inf')
    epoch = 0
    converged = False
    
    print(f"Training Stage 1 Root Neural Network (Min Epochs: {min_epochs}, Ceiling: {max_epochs})...")
    
    while epoch < max_epochs:
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred_y = model(batch_x)
            loss = criterion(pred_y, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            
        current_loss = epoch_loss / sample_size
        epoch += 1
        print(f"  Epoch {epoch} - Loss: {current_loss:.6f}")
        
        if epoch >= min_epochs:
            loss_delta = prev_loss - current_loss
            
            if 0 <= loss_delta < loss_threshold:
                print(f"  [Adaptive Output] Converged at epoch {epoch}. Stopping training loop.")
                converged = True
                break
            elif epoch == 8 and loss_delta > 1e-4:
                print("  [Adaptive Output] High gradient velocity at epoch 8. Extending budget dynamically.")
                
        prev_loss = current_loss

    if not converged and epoch >= max_epochs:
        print(f"  [WARNING] Hit absolute safety ceiling of {max_epochs} epochs without full convergence.")
        
    model.eval()
    print("Evaluating Stage 1 model to assign leaf buckets...")
    # Assign buckets in float64. The C++ runtime (rmi_benchmark.cpp) evaluates
    # Stage 1 in double precision against float64-exported weights, so doing this
    # pass in float32 makes Python and C++ disagree on the leaf index for keys
    # whose y*M lands near a bucket boundary. Those keys then get looked up
    # against a leaf whose error bound does not cover them, and the bounded
    # last-mile search misses. Observed at 0.015% of keys on SOSD books.
    model_fp64 = model.double()
    predictions = []
    with torch.no_grad():
        for i in range(0, N, 1000000):
            batch_keys = scaled_keys[i:i+1000000]
            batch_tensor = torch.tensor(batch_keys, dtype=torch.float64).unsqueeze(1).to(device)
            batch_pred = model_fp64(batch_tensor).cpu().numpy().flatten()
            predictions.append(batch_pred)
            
    y_pred = np.concatenate(predictions)
    leaf_indices = np.clip(np.floor(y_pred * M), 0, M - 1).astype(np.int64)
    
    print("Grouping keys into leaf buckets and fitting linear regressions...")
    slopes = np.zeros(M, dtype=np.float64)
    intercepts = np.zeros(M, dtype=np.float64)
    errors = np.zeros(M, dtype=np.uint64)
    bucket_counts = np.zeros(M, dtype=np.int64)
    
    for i in range(M):
        mask = (leaf_indices == i)
        bucket_keys = keys[mask]
        bucket_pos = positions[mask]
        bucket_counts[i] = len(bucket_keys)
        
        if len(bucket_keys) == 0:
            continue
        elif len(bucket_keys) == 1:
            slopes[i] = 0.0
            intercepts[i] = float(bucket_pos[0])
            errors[i] = 0
        else:
            x_raw = bucket_keys.astype(np.float64)
            y = bucket_pos.astype(np.float64)
            
            x_mean = np.mean(x_raw)
            y_mean = np.mean(y)
            
            num = np.sum((x_raw - x_mean) * (y - y_mean))
            den = np.sum((x_raw - x_mean) ** 2)
            
            m = 0.0 if den == 0 else num / den
            c = y_mean - m * x_mean
            
            slopes[i] = m
            intercepts[i] = c
            
            pred_pos = m * x_raw + c
            abs_errors = np.abs(pred_pos - y)
            errors[i] = int(np.ceil(np.max(abs_errors)))
            
    empty_buckets = np.where(bucket_counts == 0)[0]
    if len(empty_buckets) > 0:
        print(f"Interpolating parameters for {len(empty_buckets)} empty buckets...")
        valid_indices = np.where(bucket_counts > 0)[0]
        if len(valid_indices) > 0:
            for idx in empty_buckets:
                nearest_idx = valid_indices[np.argmin(np.abs(valid_indices - idx))]
                slopes[idx] = slopes[nearest_idx]
                intercepts[idx] = intercepts[nearest_idx]
                errors[idx] = errors[nearest_idx]
                
    fc1_weight = model.fc1.weight.data.cpu().numpy().flatten().astype(np.float64)
    fc1_bias = model.fc1.bias.data.cpu().numpy().flatten().astype(np.float64)
    fc2_weight = model.fc2.weight.data.cpu().numpy().flatten().astype(np.float64)
    fc2_bias = model.fc2.bias.data.cpu().numpy().flatten().astype(np.float64)
    
    param_path = os.path.join(out_dir, "rmi_params.bin")
    print(f"Exporting RMI parameters to {param_path}...")
    with open(param_path, 'wb') as f:
        f.write(np.uint64(min_key).tobytes())
        f.write(np.uint64(max_key).tobytes())
        f.write(np.uint64(N).tobytes())
        f.write(np.uint64(M).tobytes())
        
        f.write(fc1_weight.tobytes())
        f.write(fc1_bias.tobytes())
        f.write(fc2_weight.tobytes())
        f.write(fc2_bias.tobytes())
        
        f.write(slopes.tobytes())
        f.write(intercepts.tobytes())
        f.write(errors.tobytes())
        
    print(f"Successfully trained and exported RMI parameters! File size: {os.path.getsize(param_path)} bytes.")
    
    print("Error bounds stats:")
    print(f"  Max error window: {np.max(errors)}")
    print(f"  Mean error window: {np.mean(errors):.2f}")
    print(f"  Min error window: {np.min(errors)}")
    
    with open(os.path.join(out_dir, "rmi_error_stats.txt"), "w") as f_err:
        f_err.write(f"max_error: {np.max(errors)}\n")
        f_err.write(f"mean_error: {np.mean(errors):.2f}\n")
        f_err.write(f"min_error: {np.min(errors)}\n")
    
if __name__ == "__main__":
    main()