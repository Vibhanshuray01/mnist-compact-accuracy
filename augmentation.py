import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

def get_augmented_transforms():
    return transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def visualize_augmentations():
    # Create output directory if it doesn't exist
    os.makedirs('augmentation_samples', exist_ok=True)
    
    # Original transform
    original_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load dataset
    dataset = datasets.MNIST('./data', train=True, download=True, transform=None)
    
    # Get a sample image
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('MNIST Augmentation Samples')
    
    # Show original image
    img, _ = dataset[0]
    axs[0, 0].imshow(img, cmap='gray')
    axs[0, 0].set_title('Original')
    
    # Show 5 different augmentations
    augment_transform = get_augmented_transforms()
    for i in range(5):
        img_tensor = augment_transform(img)
        img_np = img_tensor.squeeze().numpy()
        row, col = (i+1)//3, (i+1)%3
        axs[row, col].imshow(img_np, cmap='gray')
        axs[row, col].set_title(f'Augmented {i+1}')
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'augmentation_samples/augmented_samples_{timestamp}.png')
    plt.close()

if __name__ == "__main__":
    visualize_augmentations() 