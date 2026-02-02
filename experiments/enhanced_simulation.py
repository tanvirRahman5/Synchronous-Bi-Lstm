# experiments/enhanced_simulation.py
import multiprocessing as mp
import threading
import time
import json
from src.client import start_fl_client_with_metrics
from src.server import start_server_with_metrics
from collections import defaultdict

# Shared metrics dictionary
metrics_dict = {
    'global_loss': [],
    'global_accuracy': [],
    'round_times': [],
    'client_accuracies': defaultdict(list),
    'round_numbers': []
}

client_ids = [0, 1, 2, 3]

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    start_time = time.time()

    # Start server in a separate process
    server_process = mp.Process(
        target=start_server_with_metrics, 
        kwargs={"num_rounds": 5, "num_clients": 4}
    )
    server_process.start()

    # Wait for server to start
    time.sleep(5)

    # Start clients
    client_threads = []
    for client_id in client_ids:
        t = threading.Thread(target=start_fl_client_with_metrics, args=(client_id,))
        t.start()
        client_threads.append(t)

    # Join clients
    for t in client_threads:
        t.join()

    # Join server
    server_process.join()
    
    total_time = time.time() - start_time
    print(f"\n✅ Total simulation time: {total_time:.2f} seconds")
