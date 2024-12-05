#!/bin/bash

# Untargeted FGSM Attacks
# L-infinity Norm
python main.py --image_path images/n01443537_goldfish.jpeg --atk_type fgsm_linf --epsilon 0.03
# L2 Norm
python main.py --image_path images/n01443537_goldfish.jpeg --atk_type fgsm_l2 --epsilon 1.0

# Untargeted PGD Attacks
# L-infinity Norm
python main.py --image_path images/n01443537_goldfish.jpeg --atk_type pgd_linf --epsilon 0.03 --max_iterations 10
# L2 Norm
python main.py --image_path images/n01443537_goldfish.jpeg --atk_type pgd_l2 --epsilon 1.0 --max_iterations 10

# Targeted FGSM Attacks (Target Class 100)
# L-infinity Norm
python main.py --image_path images/n01443537_goldfish.jpeg --atk_type fgsm_linf --epsilon 0.03 --target 100
# L2 Norm
python main.py --image_path images/n01443537_goldfish.jpeg --atk_type fgsm_l2 --epsilon 1.0 --target 100

# Targeted PGD Attacks (Target Class 100)
# L-infinity Norm
python main.py --image_path images/n01443537_goldfish.jpeg --atk_type pgd_linf --epsilon 0.03 --max_iterations 10 --target 100
# L2 Norm
python main.py --image_path images/n01443537_goldfish.jpeg --atk_type pgd_l2 --epsilon 1.0 --max_iterations 10 --target 100