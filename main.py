# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 21:57:19 2023

@author: Sankhapani
"""

import numpy as np
import matplotlib.pyplot as plt

# Load your data from the CSV file
csv_path = 'https://raw.githubusercontent.com/sankhapanineog/q2/main/data12.csv'
data = np.genfromtxt(csv_path, delimiter=',')

# Normalize the data (optional but recommended)
data = (data - np.mean(data)) / np.std(data)

# Add labels: 0 for healthy and 1 for unhealthy
threshold = 0.5  # Set a suitable threshold for labeling as unhealthy
labels = np.zeros_like(data)
labels[data > threshold] = 1

split_ratio = 0.8
split_index = int(len(data) * split_ratio)

train_data, test_data = data[:split_index], data[split_index:]
train_labels, test_labels = labels[:split_index], labels[split_index:]

# Neural network parameters
input_size = 1
hidden_size = 4
output_size = 1
learning_rate = 0.01
epochs = 10

# Initialize weights and biases
np.random.seed(42)
weights_input_hidden = np.random.rand(input_size, hidden_size)
bias_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.rand(hidden_size, output_size)
bias_output = np.zeros((1, output_size))

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Training the neural network
for epoch in range(epochs):
    for i in range(len(train_data)):
        # Forward propagation
        hidden_output = sigmoid(np.dot(train_data[i].reshape(1, -1), weights_input_hidden) + bias_hidden)
        predicted_output = sigmoid(np.dot(hidden_output, weights_hidden_output) + bias_output)

        # Calculate loss
        loss = 0.5 * np.mean((predicted_output - train_labels[i]) ** 2)

        # Backward propagation
        output_delta = (predicted_output - train_labels[i]) * sigmoid_derivative(predicted_output)
        hidden_error = output_delta.dot(weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

        # Update weights and biases
        weights_hidden_output -= learning_rate * hidden_output.T.dot(output_delta)
        bias_output -= learning_rate * np.sum(output_delta, axis=0, keepdims=True)

        weights_input_hidden -= learning_rate * train_data[i].reshape(1, -1).T.dot(hidden_delta)
        bias_hidden -= learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

# Testing the neural network
test_hidden_output = sigmoid(np.dot(test_data.reshape(-1, 1), weights_input_hidden) + bias_hidden)
predicted_test_output = sigmoid(np.dot(test_hidden_output, weights_hidden_output) + bias_output)

# Threshold for classification
threshold = 0.5
predicted_labels = (predicted_test_output > threshold).astype(int)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(data, label='Original Data', alpha=0.5)
plt.scatter(np.arange(split_index, len(data)), test_labels, color='blue', label='Actual Labels (Test Data)')
plt.plot(np.arange(split_index, len(data)), predicted_labels, label='Predicted Labels (Unhealthy)', linestyle='dashed', color='red')
plt.axhline(y=threshold, color='gray', linestyle='--', label='Threshold')
plt.title('Health Forecasting Neural Network')
plt.xlabel('Time')
plt.ylabel('Health Status')
plt.legend()
plt.show()
