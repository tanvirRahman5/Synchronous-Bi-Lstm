# experiments/run_async_simulation.py
"""
Asynchronous FL Simulation with Staleness-Aware Aggregation

This script simulates:
- Async server (continuous aggregation)
- Async clients with simulated delays and offline scenarios
- Staleness detection and parameter rejection
- Automatic sync mechanism for stale clients
"""

import multiprocessing as mp
import threading
import time
import json
from pathlib import Path
from src.async_client import start_async_client_with_delays
from src.async_server import start_async_server


# Delay configuration for each client
# - probability: 0.0-1.0 (chance of being delayed)
# - max_delay: maximum delay in seconds
DELAY_CONFIG = {
    0: {"probability": 0.0, "max_delay": 0},      # Always on-time
    1: {"probability": 0.4, "max_delay": 2.0},    # Sometimes delayed (40%)
    2: {"probability": 0.6, "max_delay": 3.0},    # Often delayed (60%)
    3: {"probability": 0.3, "max_delay": 1.5},    # Occasionally delayed (30%)
}

client_ids = [0, 1, 2, 3]


def run_async_simulation():
    """Run async FL simulation with staleness tracking"""
    mp.set_start_method("spawn", force=True)
    
    print("╔════════════════════════════════════════════════════════════╗")
    print("║   ASYNCHRONOUS FEDERATED LEARNING SIMULATION               ║")
    print("║   Staleness-Aware Aggregation with Client Delays          ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    
    # Create results directory
    results_dir = Path("experiments/results/async")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("📋 CLIENT DELAY CONFIGURATION:")
    for client_id, config in DELAY_CONFIG.items():
        prob = config["probability"] * 100
        max_d = config["max_delay"]
        print(f"   Client {client_id}: {prob:>3.0f}% chance of {max_d}s delay")
    print()
    
    # Start server in separate process
    print("🔧 Starting Server...")
    server_process = mp.Process(
        target=start_async_server,
        kwargs={
            "num_rounds": 5,
            "num_clients": 4,
            "staleness_threshold": 2,  # Reject updates if >2 rounds stale
        }
    )
    server_process.start()
    
    # Wait for server to initialize
    print("   Waiting for server to start...")
    time.sleep(5)
    
    # Start clients with delay simulation
    print("\n👥 Starting Clients...")
    client_threads = []
    
    for client_id in client_ids:
        delay_config = DELAY_CONFIG.get(client_id, {"probability": 0.0, "max_delay": 0})
        
        t = threading.Thread(
            target=start_async_client_with_delays,
            args=(client_id, delay_config),
            daemon=False
        )
        t.start()
        client_threads.append(t)
        print(f"   Client {client_id} started")
        time.sleep(0.5)  # Stagger client starts
    
    # Wait for all clients to complete
    print("\n⏳ Running simulation...")
    for t in client_threads:
        t.join()
    
    # Wait for server to finish
    server_process.join()
    
    print("\n✅ Simulation completed!")
    print(f"📁 Results saved to {results_dir}/")
    
    return results_dir


if __name__ == "__main__":
    start_time = time.time()
    results_dir = run_async_simulation()
    elapsed_time = time.time() - start_time
    
    print(f"\n⏱️  Total execution time: {elapsed_time:.2f} seconds")
