import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from typing import Dict, Tuple

class VehicleClient(fl.client.NumPyClient):
    def __init__(self, train_loader, test_loader, device):
        self.device = device
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 10)  # Adjust for KITTI classes
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

    def train(self, epochs: int = 1):
        self.model.train()
        for _ in range(epochs):
            for images, targets in self.train_loader:
                images = images.to(self.device)
                labels = targets["annotation"]["object"].to(self.device)  # Adjust based on dataset format
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

    def evaluate(self) -> Tuple[float, float]:
        self.model.eval()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for images, targets in self.test_loader:
                images = images.to(self.device)
                labels = targets["annotation"]["object"].to(self.device)  # Adjust based on dataset format
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return loss / len(self.test_loader), accuracy

    def get_parameters(self) -> Dict:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train(epochs=1)
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.evaluate()
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}
