import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTModel
from datetime import datetime
import os
from augmentation import get_augmented_transforms, visualize_augmentations

def train():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST dataset
    transform = get_augmented_transforms()
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    model.train()
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Batch: {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {100.*correct/total:.2f}%')
    
    final_accuracy = 100. * correct / total
    
    # Save model with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'mnist_model_{timestamp}.pth'
    torch.save(model.state_dict(), save_path)
    
    visualize_augmentations()
    
    return final_accuracy, save_path

if __name__ == '__main__':
    accuracy, model_path = train()
    print(f'Training completed. Final accuracy: {accuracy:.2f}%')
    print(f'Model saved at: {model_path}') 