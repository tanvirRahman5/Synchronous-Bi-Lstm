# experiments/run_simulation.py
import multiprocessing as mp
import threading
import time
from src.sync_client import start_fl_client
from src.sync_server import start_server

client_ids = [0, 1, 2, 3]

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Start server in a separate process (required for signal handlers)
    server_process = mp.Process(target=start_server, kwargs={"num_rounds": 5, "num_clients": 4})
    server_process.start()

    # Wait for server to start properly
    time.sleep(5)

    # Start clients
    client_threads = []
    for client_id in client_ids:
        t = threading.Thread(target=start_fl_client, args=(client_id,))
        t.start()
        client_threads.append(t)

    # Join clients
    for t in client_threads:
        t.join()

    # Join server
    server_process.join()
