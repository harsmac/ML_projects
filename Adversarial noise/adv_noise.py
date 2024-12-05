"""
This module provides functionality for generating adversarial examples using various attack methods.

The module implements common adversarial attack algorithms including PGD and FGSM with both L2 and
L-infinity norm constraints. It supports both targeted and untargeted attacks against pre-trained
image classification models.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

class AdversarialNoiseGenerator:
    """
    A class for generating adversarial examples using various attack methods.
    
    This class implements several adversarial attack algorithms including Projected Gradient Descent (PGD)
    and Fast Gradient Sign Method (FGSM) with both L2 and L-infinity norm constraints. It supports both
    targeted and untargeted attacks against pre-trained image classification models.
    
    Attributes:
        device (str): Device to run computations on ('cuda' or 'cpu')
        model (nn.Module): The target model to attack
        preprocess (transforms.Compose): Image preprocessing pipeline
        loss_fn (nn.Module): Loss function used for optimization
        
    Args:
        model_name (str): Name of the pre-trained model to use ('resnet50', 'resnet18', or 'resnet34')
        device (str): Device to run computations on (defaults to 'cuda' if available, else 'cpu')
    """
    
    def __init__(self, model_name='resnet50', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Load pre-trained model using the new weights parameter
        model_weights = {
            'resnet50': models.ResNet50_Weights.IMAGENET1K_V1,
            'resnet18': models.ResNet18_Weights.IMAGENET1K_V1,
            'resnet34': models.ResNet34_Weights.IMAGENET1K_V1,
        }
        
        if model_name in model_weights:
            self.model = getattr(models, model_name)(weights=model_weights[model_name])
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        # add normalization layer to model
        self.model = nn.Sequential(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225]),
            self.model
        )
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Standard image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        
        self.loss_fn = nn.CrossEntropyLoss()
        
    def generate_adversarial(self, image_path, target_class=None, atk_type='pgd_linf', epsilon=0.007, max_iterations=100):
        if atk_type not in ['pgd_l2', 'pgd_linf', 'fgsm_linf', 'fgsm_l2']:
            raise ValueError("Invalid attack type. Choose one of: 'pgd_l2', 'pgd_linf', 'fgsm_linf', 'fgsm_l2'")
        
        self.sign = -1 if target_class is not None else 1  # Positive for targeted, negative for untargeted
        
        self.targetted = True if target_class is not None else False
        self.target_class = torch.tensor([target_class]) if target_class is not None else torch.tensor([self.get_prediction(image_path)])
        
        # print(f"Original label: {self.original_label.item()}")
        # print(f"Target label: {self.target_class.item() if self.target_class is not None else None}")
        
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        x = self.preprocess(image).unsqueeze(0).to(self.device)
            
        attack_methods = {
            'pgd_l2': self.pgd_l2,
            'pgd_linf': self.pgd_linf,
            'fgsm_linf': self.fgsm_linf,
            'fgsm_l2': self.fgsm_l2
        }
        return attack_methods[atk_type](x)
    

    def pgd_l2(self, x):
        """Projected Gradient Descent with L2 norm constraint."""
        alpha = 0.5  # Step size
        x_adv = x.clone().detach().to(self.device)

        pbar = tqdm(range(self.max_iterations), desc='PGD-L2 Attack')
        for _ in pbar:
            x_adv.requires_grad_(True)

            outputs = self.model(x_adv)
            pred = outputs.argmax(dim=1)
            
            # Check if attack succeeded
            if self.targetted:
                if pred.item() == self.target_class:
                    pbar.set_description(f'PGD-L2 Attack - Target achieved at iteration {_}')
                    break
            elif pred.item() != self.target_class:
                pbar.set_description(f'PGD-L2 Attack - Misclassification achieved at iteration {_}')
                break

            # Calculate loss
            loss = self.loss_fn(outputs, self.target_class)
            
            loss.backward()

            with torch.no_grad():
                grad = x_adv.grad
                grad_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1, keepdim=True).view(-1, 1, 1, 1)
                scaled_grad = grad / (grad_norm + 1e-10)
                x_adv = x_adv + self.sign * alpha * scaled_grad
                
                # Project the perturbation to the L2 epsilon ball
                delta = x_adv - x
                delta = delta.view(delta.shape[0], -1).renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(delta)
                x_adv = torch.clamp(x + delta, 0, 1)  # Clamp to valid image range
                
                x_adv = x_adv.detach()  # Detach to prevent gradient accumulation

            pbar.set_description(f'Loss: {loss.item():.4f}')

        return x_adv.detach()

    def pgd_linf(self, x):
        """Projected Gradient Descent with L infinity norm constraint"""
        # Initialize perturbation
        alpha = 2/255.
        
        x_adv = x.clone().detach().to(self.device)
        
        pbar = tqdm(range(self.max_iterations), desc='PGD-Linf Attack')
        for _ in pbar:
            x_adv.requires_grad_(True)

            outputs = self.model(x_adv)
            pred = outputs.argmax(dim=1)
            
            # Check if attack succeeded
            if self.targetted:
                if pred.item() == self.target_class:
                    pbar.set_description(f'PGD-Linf Attack - Target achieved at iteration {_}')
                    break
            elif pred.item() != self.target_class:
                pbar.set_description(f'PGD-Linf Attack - Misclassification achieved at iteration {_}')
                break
            
            loss = self.loss_fn(outputs, self.target_class)
            
            # print(f"Loss: {loss.item()}")
            loss.backward()
            
            with torch.no_grad():
                x_adv = x_adv + self.sign * alpha * x_adv.grad.sign()
                delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                
                # clamp the perturbation to ensure valid image range
                x_adv = torch.clamp(x + delta, 0, 1)
                x_adv = x_adv.detach()
            
            pbar.set_description(f'Loss: {loss.item():.4f}')
    
        return x_adv.detach()

    def fgsm_linf(self, x):
        """Fast Gradient Sign Method with L-infinity norm constraint."""
        x_adv = x.clone().detach().to(self.device)

        # Enable gradients for the input
        x_adv.requires_grad_(True)

        # Forward pass
        outputs = self.model(x_adv)
        pred = outputs.argmax(dim=1)

        # Calculate loss
        loss = self.loss_fn(outputs, self.target_class)
        
        # Compute gradients
        loss.backward()

        with torch.no_grad():
            # Add perturbation scaled by epsilon and sign of the gradient
            x_adv = x_adv + self.sign * self.epsilon * x_adv.grad.sign()
            
            # Clamp to ensure perturbation stays within valid image range
            x_adv = torch.clamp(x_adv, 0, 1)

        print(f'FGSM-Linf Attack Completed')
        return x_adv.detach()
    
    def fgsm_l2(self, x):
        """Fast Gradient Sign Method with L2 norm constraint."""
        
        x_adv = x.clone().detach().to(self.device)

        # Enable gradients for the input
        x_adv.requires_grad_(True)

        # Forward pass
        outputs = self.model(x_adv)
        pred = outputs.argmax(dim=1)

        # Calculate loss
        loss = self.loss_fn(outputs, self.target_class)
        
        # Compute gradients
        loss.backward()

        with torch.no_grad():
            # Calculate the gradient of the loss w.r.t input
            grad = x_adv.grad
            
            # Normalize gradient to L2 norm of 1
            grad_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1, keepdim=True).view(-1, 1, 1, 1)
            scaled_grad = grad / (grad_norm + 1e-10)  # Avoid division by zero
            
            # Apply perturbation constrained by L2 norm
            x_adv = x_adv + self.sign * self.epsilon * scaled_grad
            
            # Clamp to ensure the perturbed image remains in valid range
            x_adv = torch.clamp(x_adv, 0, 1)

        print(f'FGSM-L2 Attack Completed')
        return x_adv.detach()

    def get_prediction(self, image):
        '''Function to get the prediction of the model on an image'''
        
        # if image is a path, load the image and preprocess
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            image = self.preprocess(image).unsqueeze(0).to(self.device)
        elif isinstance(image, Image.Image):
            image = self.preprocess(image).unsqueeze(0).to(self.device)
        else:
            image = image.to(self.device)
        with torch.no_grad():
            output = self.model(image)
        
        return output.argmax().item()
