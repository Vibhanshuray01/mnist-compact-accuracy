import torch
import pytest
from model import MNISTModel
from torchvision import datasets, transforms
import numpy as np
from augmentation import get_augmented_transforms
import glob
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_parameters():
    model = MNISTModel()
    param_count = count_parameters(model)
    print(f"\nTotal parameters: {param_count}")
    assert param_count < 25000, f"Model has {param_count} parameters, should be less than 25000"

def test_input_output_dimensions():
    model = MNISTModel()
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"

def test_model_accuracy():
    # Load the latest model
    import glob
    import os
    
    model_files = glob.glob('mnist_model_*.pth')
    if not model_files:
        pytest.skip("No model file found")
    
    latest_model = max(model_files, key=os.path.getctime)
    model = MNISTModel()
    model.load_state_dict(torch.load(latest_model, weights_only=True))
    model.eval()
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    assert accuracy > 95, f"Model accuracy is {accuracy:.2f}%, should be > 95%"

def test_model_robustness():
    """Test model performance on augmented data"""
    model_files = glob.glob('mnist_model_*.pth')
    if not model_files:
        pytest.skip("No model file found")
    
    latest_model = max(model_files, key=os.path.getctime)
    model = MNISTModel()
    model.load_state_dict(torch.load(latest_model, weights_only=True))
    model.eval()
    
    # Test with augmented data
    augment_transform = get_augmented_transforms()
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=augment_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    augmented_accuracy = 100 * correct / total
    print(f"\nAugmented Test Accuracy: {augmented_accuracy:.2f}%")
    assert augmented_accuracy > 90, f"Model accuracy on augmented data is {augmented_accuracy:.2f}%, should be > 90%"

def test_model_confidence():
    """Test if model predictions are confident (high probability for correct class)"""
    model_files = glob.glob('mnist_model_*.pth')
    if not model_files:
        pytest.skip("No model file found")
    
    latest_model = max(model_files, key=os.path.getctime)
    model = MNISTModel()
    model.load_state_dict(torch.load(latest_model, weights_only=True))
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)
    
    confidences = []
    with torch.no_grad():
        for data, target in test_loader:
            outputs = torch.nn.functional.softmax(model(data), dim=1)
            confidence, predicted = torch.max(outputs, 1)
            confidences.extend(confidence[predicted == target].tolist())
            if len(confidences) >= 1000:  # Test on 1000 correct predictions
                break
    
    avg_confidence = np.mean(confidences)
    print(f"\nAverage confidence on correct predictions: {avg_confidence:.2f}")
    assert avg_confidence > 0.8, f"Average confidence {avg_confidence:.2f} is too low"

def test_model_inference_time():
    """Test if model inference time is within acceptable range"""
    model = MNISTModel()
    model.eval()
    
    # Prepare batch of test data
    test_input = torch.randn(100, 1, 28, 28)
    
    # Warm-up run
    with torch.no_grad():
        _ = model(test_input)
    
    # Measure inference time
    import time
    times = []
    with torch.no_grad():
        for _ in range(10):
            start_time = time.time()
            _ = model(test_input)
            end_time = time.time()
            times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    print(f"\nAverage inference time for batch of 100: {avg_time*1000:.2f}ms")
    assert avg_time < 0.1, f"Inference too slow: {avg_time*1000:.2f}ms for batch of 100"