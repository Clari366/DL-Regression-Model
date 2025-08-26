# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:CLARISSA K

### Register Number:212224230047

```python

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

X = torch.linspace(1,70,70).reshape(-1,1)

torch.manual_seed(71)
e = torch.randint(-8,9,(70,1),dtype=torch.float)

y = 2*X + 1 + e
print(y.shape)

plt.scatter(X.numpy(), y.numpy(),color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Data for Linear Regression')
plt.show()
torch.manual_seed(59)
```

### Dataset Information
Include screenshot of the generated data

<img width="778" height="600" alt="449558468-da678a21-a694-4fbc-b4b7-9a8d66d5b7fa" src="https://github.com/user-attachments/assets/15b1a9fa-f3c7-4bb0-882d-aae86fb5dfb0" />

### OUTPUT
Training Loss Vs Iteration Plot


<img width="698" height="448" alt="479420917-3ff6bc3a-293d-4416-9c54-d178cb697507" src="https://github.com/user-attachments/assets/85d67d6e-0866-4cb4-9df7-227d0734dc3b" />

Best Fit line plot

<img width="821" height="549" alt="449558550-89472abc-d3df-4489-b7d1-8d62c661d81f" src="https://github.com/user-attachments/assets/db44d711-3d1a-4b99-8be8-aa8b92f32429" />


### New Sample Data Prediction
Include your sample input and output here


<img width="617" height="75" alt="479421567-87d8a58d-4608-4136-8d33-f91ada0d760b" src="https://github.com/user-attachments/assets/d1c54364-e6e4-4323-9e8d-f99aec66d542" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
