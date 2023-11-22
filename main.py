# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 22:54:16 2023

@author: Sankhapani
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 22:48:04 2023

@author: Sankhapani
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Function to train the neural network
def train_neural_network(train_data, train_labels, learning_rate=0.01, epochs=10):
    input_size = 1
    hidden_size = 4
    output_size = 1

    # Initialize weights and biases
    np.random.seed(42)
    weights_input_hidden = np.random.rand(input_size, hidden_size)
    bias_hidden = np.zeros((1, hidden_size))
    weights_hidden_output = np.random.rand(hidden_size, output_size)
    bias_output = np.zeros((1, output_size))

    # Training the neural network
    training_loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0
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

            # Accumulate loss for the epoch
            epoch_loss += loss

        # Store average loss for the epoch
        training_loss_history.append(epoch_loss / len(train_data))

    return training_loss_history, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output

# Function to test the neural network
def test_neural_network(test_data, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    threshold = 0.5
    predicted_test_output = []
    predicted_labels = []

    for i in range(len(test_data)):
        # Forward propagation
        hidden_output = sigmoid(np.dot(test_data[i].reshape(1, -1), weights_input_hidden) + bias_hidden)
        predicted_output = sigmoid(np.dot(hidden_output, weights_hidden_output) + bias_output)

        predicted_test_output.append(predicted_output)
        predicted_labels.append(1 if predicted_output > threshold else 0)

    return np.array(predicted_labels), np.array(predicted_test_output)

# Function to calculate accuracy
def calculate_accuracy(predicted_labels, actual_labels):
    correct_predictions = np.sum(predicted_labels == actual_labels)
    total_samples = len(actual_labels)

    if total_samples == 0:
        return 0.0

    accuracy = correct_predictions / total_samples
    return accuracy

# Streamlit app
def main():
    st.title('Health Forecasting Neural Network App')

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
    learning_rate = 0.01
    epochs = 10

    # Train the neural network
    st.subheader('Training the Neural Network')
    with st.spinner('Training in progress...'):
        training_loss_history, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output = train_neural_network(
            train_data, train_labels, learning_rate, epochs)

    st.success('Training complete!')

    # Test the neural network
    st.subheader('Testing the Neural Network')
    with st.spinner('Testing in progress...'):
        predicted_labels, predicted_test_output = test_neural_network(
            test_data, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)

    st.success('Testing complete!')

    # Display metrics
    st.subheader('Metrics')

    # Calculate and display accuracy
    accuracy = calculate_accuracy(predicted_labels, test_labels)
    st.write(f'Accuracy: {accuracy * 100:.2f}%')

    # Generate performance matrix
    true_positive = np.sum((predicted_labels == 1) & (test_labels == 1))
    false_positive = np.sum((predicted_labels == 1) & (test_labels == 0))
    true_negative = np.sum((predicted_labels == 0) & (test_labels == 0))
    false_negative = np.sum((predicted_labels == 0) & (test_labels == 1))

    # Print performance matrix
    st.write(f'True Positive: {true_positive}')
    st.write(f'False Positive: {false_positive}')
    st.write(f'True Negative: {true_negative}')
    st.write(f'False Negative: {false_negative}')

    # Calculate Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean((predicted_test_output - test_labels) ** 2))
    st.write(f'Root Mean Square Error (RMSE): {rmse:.4f}')

    # Plotting
    st.subheader('Plotting')
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Original Data', alpha=0.5)
    plt.scatter(np.arange(split_index, len(data)), test_labels, color='blue', label='Actual Labels (Test Data)')
    plt.plot(np.arange(split_index, len(data)), predicted_labels, label='Predicted Labels (Unhealthy)', linestyle='dashed', color='red')
    plt.axhline(y=threshold, color='gray', linestyle='--', label='Threshold')
    plt.title('Health Forecasting Neural Network')
    plt.xlabel('Time')
    plt.ylabel('Health Status')
    plt.legend()

    st.pyplot(plt)

if __name__ == '__main__':
    main()
