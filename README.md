# MNIST Compact Accuracy Model 


[![ML Pipeline](https://github.com/Vibhanshuray01/mnist-compact-accuracy/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/Vibhanshuray01/mnist-compact-accuracy/actions/workflows/ml-pipeline.yml)


This repository features a lightweight deep learning model for the MNIST dataset. The model is optimized to:  
1. Use **fewer than 6300 parameters**.  
2. Achieve **95% or higher training accuracy in a single epoch**.  

Additionally, GitHub Actions are set up to validate these criteria with each update.  

We have added 3 specific tests for this 
1.test_model_robustness
Augmented Test Accuracy: 91.66%
PASSED

2.test_model_confidence
Average confidence on correct predictions: 0.95
PASSED

3.test_model_inference_time
Average inference time for batch of 100: 5.87ms
PASSED

---

## Features  
- **Compact Architecture**: A well-optimized model architecture with less than 6300 parameters.  
- **High Accuracy**: Demonstrates over 95% training accuracy in just one epoch.  
- **CI/CD Pipeline**: Automated checks using GitHub Actions to verify compliance with constraints.  

---

## Requirements  

- Python 3.8 or later  
- PyTorch  
- numpy  
- matplotlib  

Install the dependencies using:  
pip install -r requirements.txt

Usage
Clone the repository:
git clone https://github.com/<your-username>/mnist-compact-accuracy.git
cd mnist-compact-accuracy

Train the model:
python train.py

View the results:
Training logs and accuracy metrics will be displayed in the terminal.

Testing via GitHub Actions
The repository is configured with GitHub Actions to ensure compliance with the following tests:

Parameter Count: Ensures the model has fewer than 6300 parameters.
Training Accuracy: Validates that the model achieves over 95% training accuracy in a single epoch.
GitHub Actions will run automatically upon pushing updates to the repository.

## Contributing
Contributions are welcome!

Fork the repository.
Create a new branch for your feature or fix.
Submit a pull request with a detailed explanation of changes.

Acknowledgments
The MNIST dataset: https://ossci-datasets.s3.amazonaws.com MNIST Database
PyTorch: https://pytorch.org/




