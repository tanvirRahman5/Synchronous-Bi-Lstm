# src/client.py
import flwr as fl
import torch
from torch.utils.data import DataLoader
from src.dataset import get_dataloader
from src.model import BiLSTMModel


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, input_size=1, num_classes=16, batch_size=32):
        self.client_id = client_id
        self.model = BiLSTMModel(input_size=input_size, hidden_size=32, num_layers=1, num_classes=num_classes)
        
        # Update path to match preprocessed data
        npz_path = f"data/partitions/client_{client_id}.npz"
        self.dataloader = get_dataloader(npz_path, batch_size=batch_size)
    
    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        # Simple training loop for demonstration
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()
        for X, y in self.dataloader:
            optimizer.zero_grad()
            output = self.model(X)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
        return self.get_parameters(), len(self.dataloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in self.dataloader:
                output = self.model(X)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return float(total - correct), len(self.dataloader.dataset), {"accuracy": correct / total}


def start_fl_client(client_id):
    client = FlowerClient(client_id)
    fl.client.start_client(server_address="localhost:8080", client=client.to_client())


def start_fl_client_with_metrics(client_id):
    """Client with metrics tracking (same as start_fl_client)"""
    client = FlowerClient(client_id)
    fl.client.start_client(server_address="localhost:8080", client=client.to_client())
