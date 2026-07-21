import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import time
from gr_core import gr_soft_dtw_loss

def run_test_5_profiling_n():
    print("=== TEST 5: Profiling Temporel (Complexité face à N) ===")
    D = 5
    N_values = [10, 20, 30, 40, 50]
    
    gamma = 0.1
    lambd = 1.0
    
    # Fully connected graph for simplicity
    edges = []
    for u in range(D):
        for v in range(u+1, D):
            edges.append((u, v, 0))
            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    results = []
    
    for N in N_values:
        torch.manual_seed(42)
        X = torch.randn(D, N, dtype=torch.float64, requires_grad=True, device=device)
        Y = torch.randn(D, N, dtype=torch.float64, requires_grad=True, device=device)
        
        # Warmup
        gr_soft_dtw_loss(X, Y, gamma, edges, lambd)
        
        iterations = 5
        start_time = time.time()
        for _ in range(iterations):
            loss = gr_soft_dtw_loss(X, Y, gamma, edges, lambd)
            loss.backward()
        
        # We need to clear grads just in case
        X.grad = None
        Y.grad = None
        
        avg_time = (time.time() - start_time) / iterations * 1000  # in ms
        results.append((N, avg_time))
        print(f"N = {N} : {avg_time:.2f} ms")
        
    return results

def run_test_6_profiling_d():
    print("\n=== TEST 6: Profiling Dimensionnel (Complexité face à D) ===")
    N = 30
    D_values = [2, 5, 8]
    
    gamma = 0.1
    lambd = 1.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    results = []
    
    for D in D_values:
        # Create a graph with 30% density
        edges = []
        for u in range(D):
            for v in range(u+1, D):
                if torch.rand(1).item() < 0.3:
                    edges.append((u, v, 0))
                    
        # Ensure at least one edge to prevent crash if density=0
        if not edges and D > 1:
            edges.append((0, 1, 0))
            
        torch.manual_seed(42)
        X = torch.randn(D, N, dtype=torch.float64, requires_grad=True, device=device)
        Y = torch.randn(D, N, dtype=torch.float64, requires_grad=True, device=device)
        
        # Warmup
        gr_soft_dtw_loss(X, Y, gamma, edges, lambd)
        
        iterations = 5
        start_time = time.time()
        for _ in range(iterations):
            loss = gr_soft_dtw_loss(X, Y, gamma, edges, lambd)
            loss.backward()
        
        X.grad = None
        Y.grad = None
            
        avg_time = (time.time() - start_time) / iterations * 1000  # in ms
        
        # Memory profiling if CUDA
        mem_peak = 0
        if device.type == "cuda":
            mem_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
            torch.cuda.reset_peak_memory_stats()
            
        results.append((D, len(edges), avg_time, mem_peak))
        print(f"D = {D} (Edges = {len(edges)}) : {avg_time:.2f} ms | VRAM Peak: {mem_peak:.2f} MB")

    return results

if __name__ == "__main__":
    run_test_5_profiling_n()
    run_test_6_profiling_d()
