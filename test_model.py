import torch
import pytest
from model import MNISTModel
from torchvision import datasets, transforms

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