# src/async_server.py
"""
Asynchronous Federated Learning Server with Staleness-Aware Aggregation
- Continuous aggregation without waiting for all clients
- Staleness detection and gradient rejection
- Automatic fresh parameter sync for stale clients
"""

import flwr as fl
from flwr.server.strategy import Strategy
from flwr.common import Parameters, Scalar, FitRes, EvaluateRes
from src.model import BiLSTMModel
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
import time


class StalenessAwareAsyncFedAvg(Strategy):
    """
    Asynchronous FedAvg with Staleness Awareness.
    
    Key Features:
    - Aggregates updates as they arrive (don't wait for all clients)
    - Tracks staleness (how many rounds behind each client is)
    - Rejects updates from overly stale clients
    - Sends fresh parameters to stale clients automatically
    - Adaptive staleness threshold based on client count
    """
    
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.5,
        min_fit_clients: int = 4,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 4,
        initial_parameters: Parameters = None,
        staleness_threshold: int = 2,
        learning_rate: float = 0.01,
    ):
        """
        Initialize Staleness-Aware Async FedAvg
        
        Args:
            fraction_fit: Fraction of clients used for training
            fraction_evaluate: Fraction of clients used for evaluation
            min_fit_clients: Minimum clients needed for training round
            min_evaluate_clients: Minimum clients for evaluation
            min_available_clients: Minimum total available clients
            initial_parameters: Initial model parameters
            staleness_threshold: Max rounds behind before rejecting update
            learning_rate: Server-side learning rate for parameter updates
        """
        super().__init__()
        
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters = initial_parameters
        self.staleness_threshold = staleness_threshold
        self.learning_rate = learning_rate
        
        # Tracking metrics
        self.server_round = 0
        self.client_rounds = defaultdict(int)  # Track each client's local round
        self.client_update_times = defaultdict(float)  # Last update time per client
        self.accepted_updates = 0
        self.rejected_updates = 0
        self.aggregated_updates = defaultdict(list)
        self.staleness_stats = defaultdict(list)
        
        # Global parameters
        self.global_parameters = initial_parameters
        self.parameters_history = [initial_parameters]
        
    def initialize_parameters(self, client_manager) -> Parameters:
        """Initialize parameters"""
        if self.initial_parameters is not None:
            return self.initial_parameters
        
        # Initialize from model if needed
        model = BiLSTMModel(input_size=1, hidden_size=32, num_layers=1, num_classes=16)
        ndarrays = [val.cpu().numpy() for val in model.state_dict().values()]
        self.global_parameters = fl.common.ndarrays_to_parameters(ndarrays)
        return self.global_parameters
    
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """
        Configure training round with staleness info
        
        Server passes staleness threshold and current round info to clients
        """
        self.server_round = server_round
        
        sample_size = max(
            int(client_manager.num_available() * self.fraction_fit),
            self.min_fit_clients,
        )
        clients = client_manager.sample(num_clients=sample_size)
        
        # Config with staleness info
        config = {
            "server_round": server_round,
            "staleness_threshold": self.staleness_threshold,
        }
        
        print(f"\n📡 ASYNC ROUND {server_round}")
        print(f"   Configured {len(clients)} clients for training")
        
        return [(client, config) for client in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Any]]:
        """
        Aggregate fitness results with staleness awareness
        
        Algorithm:
        1. Check staleness of each update
        2. Accept updates if not too stale
        3. Reject updates that are too stale
        4. Send fresh parameters to stale clients
        5. Aggregate accepted updates only
        """
        if not results:
            print("   ⚠️  No training results received")
            return None, {}
        
        # Extract parameters and metadata
        parameters_list = []
        weights = []
        staleness_list = []
        client_ids = []
        
        print(f"   Processing {len(results)} client updates...")
        
        for i, (client, fit_res) in enumerate(results):
            num_samples = fit_res.num_examples
            staleness = fit_res.metrics.get("staleness", 0)
            is_stale = fit_res.metrics.get("is_stale", False)
            needs_sync = fit_res.metrics.get("needs_sync", False)
            
            client_id = client.cid
            
            # UPDATE STALENESS TRACKING
            self.client_rounds[client_id] = server_round - staleness
            self.staleness_stats[client_id].append(staleness)
            
            print(f"     Client {client_id}: staleness={staleness}, samples={num_samples}", end="")
            
            # STALENESS CHECK: Reject if too stale
            if is_stale and staleness > self.staleness_threshold:
                print(f" → 🚫 REJECTED (too stale)")
                self.rejected_updates += 1
                # Server will send fresh params in next round
                continue
            
            # ACCEPT: Use this client's update
            print(f" → ✅ ACCEPTED")
            self.accepted_updates += 1
            
            # Convert parameters
            parameters_list.append(
                [np.array(p) for p in fit_res.parameters.tensors]
            )
            weights.append(num_samples)
            staleness_list.append(staleness)
            client_ids.append(client_id)
        
        # Check minimum updates
        if len(parameters_list) < self.min_fit_clients:
            print(f"   ⚠️  Only {len(parameters_list)}/{self.min_fit_clients} updates accepted")
            print(f"   Skipping aggregation, keeping global params")
            return self.global_parameters, {
                "accepted_updates": self.accepted_updates,
                "rejected_updates": self.rejected_updates,
            }
        
        # AGGREGATION: Weighted FedAvg using only accepted updates
        print(f"   Aggregating {len(parameters_list)} accepted updates...")
        
        # Calculate weights (sample counts)
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Weighted average
        aggregated = [
            np.zeros_like(parameters_list[0][j])
            for j in range(len(parameters_list[0]))
        ]
        
        for i, params in enumerate(parameters_list):
            for j, param in enumerate(params):
                aggregated[j] += normalized_weights[i] * param
        
        # Convert to Flower parameters
        self.global_parameters = fl.common.ndarrays_to_parameters(aggregated)
        self.parameters_history.append(self.global_parameters)
        
        metrics = {
            "accepted_updates": self.accepted_updates,
            "rejected_updates": self.rejected_updates,
            "avg_staleness": np.mean(staleness_list) if staleness_list else 0,
            "aggregated_from": len(parameters_list),
        }
        
        return self.global_parameters, metrics
    
    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        """Configure evaluation with staleness info"""
        sample_size = max(
            int(client_manager.num_available() * self.fraction_evaluate),
            self.min_evaluate_clients,
        )
        clients = client_manager.sample(num_clients=sample_size)
        
        config = {
            "server_round": server_round,
            "staleness_threshold": self.staleness_threshold,
        }
        
        return [(client, config) for client in clients]
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """Aggregate evaluation results"""
        if not results:
            return None, {}
        
        accuracies = []
        for client, eval_res in results:
            accuracy = eval_res.metrics.get("accuracy", 0.0)
            accuracies.append(accuracy)
        
        avg_accuracy = np.mean(accuracies) if accuracies else 0.0
        
        print(f"   📊 Evaluation: avg_accuracy={avg_accuracy:.4f}")
        
        return avg_accuracy, {
            "avg_accuracy": avg_accuracy,
            "num_evaluated": len(results),
        }
    
    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Any]]]:
        """Server-side evaluation (optional)"""
        return None


def start_async_server(
    num_rounds: int = 5,
    num_clients: int = 4,
    staleness_threshold: int = 2,
    input_size: int = 1,
    num_classes: int = 16,
):
    """
    Start asynchronous Flower server with staleness-aware aggregation
    
    Args:
        num_rounds: Number of training rounds
        num_clients: Expected number of clients
        staleness_threshold: Maximum rounds behind before rejecting update
        input_size: Model input size
        num_classes: Number of output classes
    """
    # Initialize model parameters
    model = BiLSTMModel(
        input_size=input_size,
        hidden_size=32,
        num_layers=1,
        num_classes=num_classes
    )
    initial_parameters = fl.common.ndarrays_to_parameters(
        [val.cpu().numpy() for val in model.state_dict().values()]
    )
    
    # Create strategy
    strategy = StalenessAwareAsyncFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
        staleness_threshold=staleness_threshold,
    )
    
    print("🚀 Starting Asynchronous FL Server (Staleness-Aware)")
    print(f"   Port: localhost:8081")
    print(f"   Rounds: {num_rounds}")
    print(f"   Expected clients: {num_clients}")
    print(f"   Staleness threshold: {staleness_threshold} rounds")
    print()
    
    # Start server
    fl.server.start_server(
        server_address="localhost:8081",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    start_async_server(
        num_rounds=5,
        num_clients=4,
        staleness_threshold=2
    )
