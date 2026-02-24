#!/usr/bin/env python3
"""
Test F.linear performance with various shapes.
Input shape: [M, K], Weight shape: [N, K], Output shape: [M, N]
All tensors are bf16 dtype.

Note: Using cudaEvent for timing to match C++ implementation.
"""

import torch
import torch.nn.functional as F
import json

# Test configurations
M_LIST = [1, 7, 143, 256, 8192, 32000]
K_LIST = [4096, 4096 * 3, 16384]
N_LIST = [4096, 8192, 12288]

WARMUP_ITER = 10
TEST_ITER = 100


def test_linear(M, K, N, device="cuda"):
    """Test F.linear performance for given shape.
    
    Note: Using torch.cuda.Event for timing to match C++ implementation.
    This measures pure GPU execution time without CPU overhead.
    """
    # Create tensors with RANDOM initialization
    input_tensor = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    weight = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    
    # Warmup - more iterations for better stability
    for _ in range(WARMUP_ITER * 2):
        output = F.linear(input_tensor, weight)
    torch.cuda.synchronize()
    
    # Small delay to let GPU stabilize
    torch.cuda._sleep(1000000)  # 1ms in nanoseconds
    
    # Test using cudaEvent (matches C++ implementation)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(TEST_ITER):
        output = F.linear(input_tensor, weight)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end) / TEST_ITER
    
    # Calculate TFLOPS
    flops = 2 * M * N * K
    tflops = flops / (elapsed_ms * 1e-3) / 1e12
    
    return {
        "M": M,
        "K": K,
        "N": N,
        "time_ms": elapsed_ms,
        "tflops": tflops,
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    device = "cuda"
    print(f"Testing on {torch.cuda.get_device_name(device)}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Timing method: torch.cuda.Event (matches C++)")
    print()
    
    results = []
    
    print(f"{'M':>8} {'K':>8} {'N':>8} {'Time(ms)':>12} {'TFLOPS':>12}")
    print("-" * 60)
    
    for M in M_LIST:
        for K in K_LIST:
            for N in N_LIST:
                try:
                    result = test_linear(M, K, N, device)
                    results.append(result)
                    print(f"{M:>8} {K:>8} {N:>8} {result['time_ms']:>12.3f} {result['tflops']:>12.2f}")
                except RuntimeError as e:
                    print(f"{M:>8} {K:>8} {N:>8} ERROR: {e}")
    
    print()
    print("=" * 60)
    print("Summary by M:")
    print("-" * 60)
    
    for M in M_LIST:
        m_results = [r for r in results if r["M"] == M]
        if m_results:
            avg_tflops = sum(r["tflops"] for r in m_results) / len(m_results)
            max_tflops = max(r["tflops"] for r in m_results)
            print(f"M={M:>6}: avg={avg_tflops:>10.2f} TFLOPS, max={max_tflops:>10.2f} TFLOPS")
    
    print()
    print("Summary by K:")
    print("-" * 60)
    
    for K in K_LIST:
        k_results = [r for r in results if r["K"] == K]
        if k_results:
            avg_tflops = sum(r["tflops"] for r in k_results) / len(k_results)
            max_tflops = max(r["tflops"] for r in k_results)
            print(f"K={K:>6}: avg={avg_tflops:>10.2f} TFLOPS, max={max_tflops:>10.2f} TFLOPS")
    
    # Save results to JSON
    with open("test_linear_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print()
    print("Results saved to test_linear_results.json")


if __name__ == "__main__":
    main()
