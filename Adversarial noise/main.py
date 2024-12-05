import matplotlib.pyplot as plt
from PIL import Image
import torch
import argparse
from adv_noise import *
from utils import *

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate adversarial images')
    parser.add_argument('--image_path', type=str, default='images/n01843383_toucan.jpeg', help='Path to input image')
    parser.add_argument('--atk_type', type=str, default='pgd_linf', help='Attack type (default: pgd_linf)')
    parser.add_argument('--epsilon', type=float, default=0.03, help='Epsilon value for attack (default: 0.007)')
    parser.add_argument('--max_iterations', type=int, default=10, help='Maximum iterations (default: 100)')
    parser.add_argument('--target_class', type=int, default=None, help='Target class for attack (default: None)')
    args = parser.parse_args()

    # Initialize the generator
    generator = AdversarialNoiseGenerator()
    # print(f"Using device: {generator.device}")

    try:
        # Step 1. Load original image
        original_image = preprocess_image(args.image_path, generator.device)
        
        # Step 2. Generate adversarial image
        adversarial_image = generator.generate_adversarial(
            args.image_path, 
            target_class=args.target_class,
            atk_type=args.atk_type,
            epsilon=args.epsilon,
            max_iterations=args.max_iterations
            )
        
        # Step 3. Get and display classification results
        orig_pred, target_pred = get_classification_results(original_image, adversarial_image, generator)
            
        # Step 4. Plot the images
        plot_images(args, original_image, adversarial_image, orig_pred, target_pred)
        
    except FileNotFoundError:
        print(f"Error: File not found at {args.image_path}. Please provide a valid image path.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()