# src/async_client.py
"""
Asynchronous Federated Learning Client with Staleness Tracking
- Independent operation without waiting for other clients
- Staleness detection and handling
- Automatic sync when receiving fresh global parameters
"""

import flwr as fl
import torch
import numpy as np
import time
from src.dataset import get_dataloader
from src.model import BiLSTMModel


class AsyncFlowerClient(fl.client.NumPyClient):
    """
    Asynchronous FL Client with staleness-aware updates.
    
    Key Features:
    - Tracks local round number vs server round
    - Detects staleness and syncs when needed
    - Can be delayed without blocking other clients
    """
    
    def __init__(self, client_id, input_size=1, num_classes=16, batch_size=32, 
                 delay_probability=0.0, max_delay=0):
        """
        Initialize async client
        
        Args:
            client_id: Client identifier
            input_size: LSTM input feature size
            num_classes: Number of output classes
            batch_size: Training batch size
            delay_probability: Probability of being delayed (0.0-1.0)
            max_delay: Maximum delay in seconds
        """
        self.client_id = client_id
        self.model = BiLSTMModel(
            input_size=input_size, 
            hidden_size=32, 
            num_layers=1, 
            num_classes=num_classes
        )
        
        npz_path = f"data/partitions/client_{client_id}.npz"
        self.dataloader = get_dataloader(npz_path, batch_size=batch_size)
        
        # Staleness tracking
        self.local_round = 0
        self.server_round = 0
        self.last_update_round = 0
        self.staleness = 0
        self.is_stale = False
        
        # Delay simulation
        self.delay_probability = delay_probability
        self.max_delay = max_delay
        self.is_delayed = False
        self.last_global_params = None
        
    def _simulate_delay(self):
        """Simulate network delay or client being offline"""
        if np.random.random() < self.delay_probability:
            delay = np.random.uniform(0, self.max_delay)
            self.is_delayed = True
            print(f"  [CLIENT {self.client_id}] 🔄 Delaying for {delay:.2f}s...")
            time.sleep(delay)
            self.is_delayed = False
            
    def _calculate_staleness(self, server_round):
        """Calculate how many rounds behind this client is"""
        self.server_round = server_round
        self.staleness = max(0, self.server_round - self.local_round)
        self.is_stale = self.staleness > 0
        
        if self.is_stale:
            print(f"  [CLIENT {self.client_id}] ⚠️  Staleness detected: {self.staleness} rounds behind")
        
        return self.staleness
    
    def get_parameters(self):
        """Get current model parameters"""
        params = [val.cpu().numpy() for val in self.model.state_dict().values()]
        self.last_global_params = params
        return params

    def set_parameters(self, parameters):
        """Set model parameters from server"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        Async training: client trains locally without waiting
        
        config includes:
        - 'server_round': Current round on server (for staleness check)
        - 'staleness_threshold': Max allowed staleness
        """
        # Simulate potential delay
        self._simulate_delay()
        
        # Extract config
        server_round = config.get('server_round', 0)
        staleness_threshold = config.get('staleness_threshold', 2)
        
        # Calculate staleness
        self._calculate_staleness(server_round)
        
        # Check if too stale - if yes, reject and request sync
        if self.staleness > staleness_threshold:
            print(f"  [CLIENT {self.client_id}] 🚫 TOO STALE (threshold={staleness_threshold})")
            print(f"  [CLIENT {self.client_id}] 📥 Requesting fresh parameters from server...")
            # Return old params and request fresh sync
            return self.get_parameters(), len(self.dataloader.dataset), {
                "num_samples": len(self.dataloader.dataset),
                "staleness": self.staleness,
                "is_stale": True,
                "needs_sync": True
            }
        
        # Normal training path
        self.set_parameters(parameters)
        self.model.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Training loop
        total_loss = 0
        num_batches = 0
        for X, y in self.dataloader:
            optimizer.zero_grad()
            output = self.model(X)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        
        # Update local round
        self.local_round = server_round
        
        print(f"  [CLIENT {self.client_id}] ✓ Trained (Round {server_round}, Staleness: {self.staleness})")
        
        return self.get_parameters(), len(self.dataloader.dataset), {
            "num_samples": len(self.dataloader.dataset),
            "staleness": self.staleness,
            "is_stale": False,
            "avg_loss": total_loss / num_batches
        }

    def evaluate(self, parameters, config):
        """Evaluate model on local data"""
        self.set_parameters(parameters)
        self.model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for X, y in self.dataloader:
                output = self.model(X)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        accuracy = correct / total
        loss = float(total - correct)
        
        print(f"  [CLIENT {self.client_id}] 📊 Evaluated: {accuracy:.4f}")
        
        return loss, len(self.dataloader.dataset), {
            "accuracy": accuracy,
            "staleness": self.staleness
        }


def start_async_client(client_id, delay_probability=0.0, max_delay=0):
    """Start async client with optional delay simulation"""
    client = AsyncFlowerClient(
        client_id, 
        delay_probability=delay_probability,
        max_delay=max_delay
    )
    fl.client.start_client(
        server_address="localhost:8081",  # Different port for async
        client=client.to_client()
    )


def start_async_client_with_delays(client_id, delay_config):
    """Start async client with specific delay configuration"""
    delay_prob = delay_config.get('probability', 0.0)
    max_delay = delay_config.get('max_delay', 0)
    start_async_client(client_id, delay_probability=delay_prob, max_delay=max_delay)
