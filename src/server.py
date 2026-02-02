# src/server.py
import flwr as fl
from src.model import BiLSTMModel
import torch
import time

def start_server(num_rounds=5, num_clients=4, input_size=1, num_classes=16):
    # Initialize model (needed for initial parameters)
    model = BiLSTMModel(input_size=input_size, hidden_size=32, num_layers=1, num_classes=num_classes)
    
    # Convert model weights to Flower parameters
    initial_parameters = fl.common.ndarrays_to_parameters([val.cpu().numpy() for val in model.state_dict().values()])
    
    # Define strategy (FedAvg)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,            # fraction of clients used for training each round
        fraction_evaluate=0.5,       # fraction of clients used for evaluation each round
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
    )

    # Start the server
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    start_server()


def start_server_with_metrics(num_rounds=5, num_clients=4, input_size=1, num_classes=16):
    """Server with enhanced metrics tracking"""
    # Initialize model (needed for initial parameters)
    model = BiLSTMModel(input_size=input_size, hidden_size=32, num_layers=1, num_classes=num_classes)
    
    # Convert model weights to Flower parameters
    initial_parameters = fl.common.ndarrays_to_parameters([val.cpu().numpy() for val in model.state_dict().values()])
    
    # Define strategy (FedAvg)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
    )

    # Start the server
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
