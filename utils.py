import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Relu function
def relu(x):
    return np.maximum(0, x)

# Relu function changed to be applied in all matrix itens 
def relu_vectorized(x):
    return np.maximum(0, x)

# Relu function derivative
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Softmax function
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis = 1,keepdims=True)

# Cost fuction using sparse cross entropy loss 
def cost(A3, Y):
    m = Y.shape[0]
    return -np.sum(Y * np.log(A3 + 1e-8)) / m

# Creates OneHot encoding
def one_hot(y):
    n_classes = np.max(y)+1
    y_hot = np.zeros((y.shape[0],n_classes))
    for n in range(y.shape[0]):
        y_hot[n][y[n][0]]=1
    
    return y_hot

# Creates initial random weights for the first forward propagation 
def random_weights(input_size, hidden_layer1_size, hidden_layer2_size, output_size):
    weights = {
        'W1': np.random.randn(input_size, hidden_layer1_size) * 0.01,
        'b1': np.zeros((1, hidden_layer1_size)),
        'W2': np.random.randn(hidden_layer1_size, hidden_layer2_size) * 0.01,
        'b2': np.zeros((1, hidden_layer2_size)),
        'W3': np.random.randn(hidden_layer2_size, output_size) * 0.01,
        'b3': np.zeros((1, output_size)),
    }
    return weights

# Forward propagation process
def forward(X, weights, return_back='forward'):
    # Input layer
    Z1 = np.dot(X, weights['W1']) + weights['b1'] # Result before activation function
    A1 = relu_vectorized(Z1) # Result after activation function Relu
    # Second layer
    Z2 = np.dot(A1, weights['W2']) + weights['b2'] # Result before activation function
    A2 = relu_vectorized(Z2) # Result after activation function Relu
    # Output layer
    Z3 = np.dot(A2, weights['W3']) + weights['b3'] # Result before activation function
    A3 = softmax(Z3) # Result after activation function Softmax
    # Return type 1
    forward_data = (Z1, A1, Z2, A2, Z3, A3)
    if return_back == 'forward': 
        return forward_data
    # Return type 2 when using prediction, since we'll only need the A3 results to predict
    elif return_back == 'predict':
        return A3

# Backward propagation process
def back_propagation(X, Y, forward_data, weights):
    # loads the results of the last forward propagation
    Z1, A1, Z2, A2, Z3, A3 = forward_data
    # number of samples in the training
    m = X.shape[0] 

    # Process
    # Output layer
    dZ3 = A3 - Y # Derivative of cost regarding Z3 (output layer result before activation function)
    gradient_W3 = np.dot(A2.T, dZ3) / m # Gradient of W3(output layer weights matrix)
    gradient_b3 = np.sum(dZ3, axis=0, keepdims=True) / m  # Gradient of b3(output layer bias matrix)
    
    # Second layer
    dA2 = np.dot(dZ3, weights['W3'].T) # Derivative of cost regarding A2(Second layer results)
    dZ2 = dA2 * relu_derivative(Z2) # Derivative of cost regarding Z2(second layer results before activation function)
    gradient_W2 = np.dot(A1.T, dZ2) / m # Gradient of W2(second layer weights matrix) 
    gradient_b2 = np.sum(dZ2, axis=0, keepdims=True) / m # Gradient of b2(second layer bias matrix)

    # Input Layer 
    dA1 = np.dot(dZ2, weights['W2'].T) # Derivative of cost regarding A1(input layer results)
    dZ1 = dA1 * relu_derivative(Z1) # Derivative of cost regarding Z1(input layer results before activation function)
    gradient_W1 = np.dot(X.T, dZ1) / m # Gradient of W1(input layer weights matrix) 
    gradient_b1 = np.sum(dZ1, axis=0, keepdims=True) / m # Gradient of b1(input layer bias matrix)

    # Save gradients as dictionary
    gradients = {
        'gradient_W3': gradient_W3, 'gradient_b3': gradient_b3,
        'gradient_W2': gradient_W2, 'gradient_b2': gradient_b2,
        'gradient_W1': gradient_W1, 'gradient_b1': gradient_b1
    }
    return gradients

# Updates the weights using gradient descent
def update_weights(weights, gradients, learning_rate):
    weights['W3'] -= gradients['gradient_W3'] * learning_rate
    weights['b3'] -= gradients['gradient_b3'] * learning_rate
    weights['W2'] -= gradients['gradient_W2'] * learning_rate
    weights['b2'] -= gradients['gradient_b2'] * learning_rate
    weights['W1'] -= gradients['gradient_W1'] * learning_rate
    weights['b1'] -= gradients['gradient_b1'] * learning_rate

    return weights

# Trains the model using batch gradient descent.
def train(X, Y, input_size, hidden_layer1_size, hidden_layer2_size, output_size, learning_rate, epochs, batch_size, weights):
    # Inputs
    m = X.shape[0] # check number os samples in train
    Y = one_hot(Y) # transform the Y labels in to one_hot encoding format

    # Epoch process  
    for epoch in range(epochs):
        # Shuffle the data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]
        
        # Perform the gradient descent in batchs
        for i in range(0, m, batch_size):
            # Batch sample
            X_batch = X_shuffled[i:i + batch_size]
            Y_batch = Y_shuffled[i:i + batch_size]
            # Forward propagation with the batch sample
            forward_data = forward(X_batch, weights)
            # Perform backpropagation.
            gradients = back_propagation(X_batch, Y_batch, forward_data, weights)
            # wights update
            weights = update_weights(weights, gradients, learning_rate)

        # Cost calculation per epoch
        forward_data_full = forward(X, weights)
        _, _, _, _, _, A3 = forward_data_full
        epoch_cost = cost(A3, Y)
        # Print cost for each multiple of 10 epoch count
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, cost: {epoch_cost}')
    
    return weights

# Prediction function
def predict(X, weights):
    A3 = forward(X, weights, return_back = 'predict')
    return np.argmax(A3, axis=1)

# Save model function
def save_model(file_name, model):
    with open(file_name+'.pkl', 'wb') as file: 
        pickle.dump(model,file)

# Load model function
def load_model(file_name):
    with open(file_name+'.pkl', 'rb') as file:
        model_loaded = pickle.load(file)

    return model_loaded

