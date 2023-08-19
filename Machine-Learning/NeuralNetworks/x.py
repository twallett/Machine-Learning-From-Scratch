#%%
import numpy as np

# Generate synthetic data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 3 * X + 2 + 0.1 * np.random.randn(100, 1)

# Initialize weights and biases
input_size = 1
hidden_size = 64
output_size = 1
learning_rate = 0.01

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Training loop
batch_size = 32
num_batches = len(X) // batch_size

loss_ = []
output = []
for epoch in range(1000):
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = start_idx + batch_size
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]

        # Forward Propagation
        z1 = np.dot(X_batch, W1) + b1
        a1 = np.maximum(0, z1)  # ReLU activation
        z2 = np.dot(a1, W2) + b2

        # Loss calculation
        loss = np.mean((z2 - y_batch)**2)

        # Backpropagation
        dz2 = 2 * (z2 - y_batch) / batch_size
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0)
        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * (z1 > 0)
        dW1 = np.dot(X_batch.T, dz1)
        db1 = np.sum(dz1, axis=0)

        # Update weights and biases
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        
        loss_.append(loss)
        output.append(z2)

# Prediction
new_data = np.array([[0.6]])
z1_new = np.dot(new_data, W1) + b1
a1_new = np.maximum(0, z1_new)
z2_new = np.dot(a1_new, W2) + b2
predicted_value = z2_new

print("Predicted:", predicted_value)

# %%

output = np.array(output).reshape(-1,1)
# %%

import numpy as np

# Define the logistic sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate synthetic data with two features
np.random.seed(0)
X = np.random.rand(100, 2)  # Two features
y = 3 * X[:, 0:1] + 2 * X[:, 1:2] + 0.1 * np.random.randn(100, 1)

# Initialize weights and biases
input_size = 2  # Number of features
hidden_size = 64
output_size = 1
learning_rate = 0.01

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Training loop
batch_size = 32
num_batches = len(X) // batch_size


loss_ = []
output = []
for epoch in range(1000):
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = start_idx + batch_size
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]

        # Forward Propagation
        z1 = np.dot(X_batch, W1) + b1
        a1 = sigmoid(z1)  # Logistic sigmoid activation
        z2 = np.dot(a1, W2) + b2

        # Loss calculation
        loss = np.mean((z2 - y_batch)**2)

        # Backpropagation
        dz2 = 2 * (z2 - y_batch) / batch_size
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0)
        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * (a1 * (1 - a1))  # Derivative of sigmoid
        dW1 = np.dot(X_batch.T, dz1)
        db1 = np.sum(dz1, axis=0)

        # Update weights and biases
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        
        loss_.append(loss)
        output.append(z2)

# Prediction for new data with two features
new_data = np.array([[0.6, 0.7]])  # Two features
z1_new = np.dot(new_data, W1) + b1
a1_new = sigmoid(z1_new)
z2_new = np.dot(a1_new, W2) + b2
predicted_value = z2_new

print("Predicted:", predicted_value)

# %%
