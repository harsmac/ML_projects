import torch
import logging
import flwr as fl
from client_train import VehicleClient
from data import get_dataloader

def start_client(device: str):
    # Simulate train/test split
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting Flower client")
    train_loader = get_dataloader("train", batch_size=32)
    test_loader = get_dataloader("val", batch_size=32)
    client = VehicleClient(train_loader, test_loader, device)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
    logger.info("Flower client started")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_client(device)
