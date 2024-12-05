import matplotlib.pyplot as plt
from PIL import Image
import torch
import argparse
import torchvision.transforms as transforms
from torchvision.models.resnet import _IMAGENET_CATEGORIES

from adv_noise import *

"""Utility functions for adversarial image generation and visualization.

This module provides functions to:
- Preprocess images for model input
- Generate and visualize adversarial examples 
- Plot original and adversarial images side by side
- Get model predictions and class names
"""

# Define preprocessing transformation as a constant
PREPROCESS_TRANSFORM = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

def plot_images(args, original_image, adversarial_image, original_pred, adversarial_pred):
    """Plot original and adversarial images side by side with difference"""
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(original_image.squeeze(0).permute(1, 2, 0).cpu().numpy())
    plt.title(f'Original Image\nClass: {get_class_name(original_pred)}')
    
    plt.subplot(1, 4, 2)
    plt.imshow(adversarial_image.squeeze(0).permute(1, 2, 0).cpu().numpy())
    plt.title(f'Adversarial Image\nClass: {get_class_name(adversarial_pred)}')
    
    plt.subplot(1, 4, 3)
    difference = (adversarial_image.squeeze(0) - original_image.squeeze(0))
    difference = torch.clamp(difference, 0, 1)  # Clamp to valid image range
    plt.imshow(difference.permute(1, 2, 0).cpu().numpy())
    plt.title('Difference')
    
    plt.subplot(1, 4, 4)
    scaled_diff = torch.clamp(difference * 128, 0, 1)  # Scale and clamp to [0,1]
    plt.imshow(scaled_diff.permute(1, 2, 0).cpu().numpy())
    plt.title('Difference (x128)')
    
    # extract save_file_name from the original image path
    image_file_name = args.image_path.split('/')[-1].split('.')[0]
    target_class_str = 'targeted' if args.target_class is not None else 'untargeted'
    save_file_name = f'{image_file_name}_{args.atk_type}_{args.epsilon}_{args.max_iterations}_{target_class_str}'
    
    save_name = f'adversarial_images/{save_file_name}.png'
    plt.savefig(save_name)
    # plt.show()
    
    return


def get_classification_results(original_image, adversarial_image, generator):
    """Get and print classification results for both images"""
    original_pred = generator.get_prediction(original_image)
    adversarial_pred = generator.get_prediction(adversarial_image)
    
    # print("Image classification results:")
    # print(f"Original image classified as: {get_class_name(original_pred)}")
    # print(f"Adversarial image classified as: {get_class_name(adversarial_pred)}")
    
    return original_pred, adversarial_pred

def preprocess_image(image_path, device):
    image = Image.open(image_path).convert('RGB')
    return PREPROCESS_TRANSFORM(image).unsqueeze(0).to(device)

    
def get_class_name(class_idx):
    """Return the class name for a given class index.
    
    Args:
        class_idx (int): Index of the class
        
    Returns:
        str: Class name corresponding to the index
    """
    try:
        class_name = _IMAGENET_CATEGORIES[class_idx]
        # Extract just the class term before the comma if it exists
        return class_name.split(',')[0].strip()
    except ImportError:
        return str(class_idx)  # Fallback to index if names unavailable
