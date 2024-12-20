import os
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision import transforms

# Define data transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Path to KITTI or VOC-like dataset
data_dir = "./kitti"

# Load dataset
def load_kitti_data(split: str):
    dataset = VOCDetection(
        root=data_dir,
        year="2012",  # Simulating KITTI-like setup with Pascal VOC
        image_set=split,
        transform=transform,
    )
    return dataset

def get_dataloader(split: str, batch_size: int):
    dataset = load_kitti_data(split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
