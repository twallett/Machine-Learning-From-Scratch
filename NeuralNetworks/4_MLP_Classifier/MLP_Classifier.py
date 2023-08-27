#%%
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Load MNIST dataset
mnist = fetch_openml('mnist_784')
X, y = mnist.data / 255.0, mnist.target.astype(int)

X = np.array(X)
y = np.array(y).reshape(-1,1)

# Convert labels to one-hot encoding
encoder = OneHotEncoder()
y = encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train = y_train.toarray()
y_test = y_test.toarray()

#%%
def logsig(n):
    return 1 / (1 + np.exp(-n))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(pred, true):
    return -np.mean(np.sum(true * np.log(pred), axis=1))
    
def initialize_weights(layers_size):
    weights = [np.random.randn(layers_size[i], layers_size[i + 1]) for i in range(len(layers_size) - 1)]
    biases = [np.zeros((1, layers_size[i + 1])) for i in range(len(layers_size) - 1)]
    return weights, biases

def forward_propagation(inputs, weights, biases):
    activations = [inputs]
    for i in range(len(weights) - 1):
        n = np.dot(activations[i], weights[i]) + biases[i]
        a = logsig(n)
        activations.append(a)
    z_output = np.dot(activations[-1], weights[-1]) + biases[-1]
    a_output = softmax(z_output)
    activations.append(a_output)
    return activations

def MLP_Classifier_Train(inputs, targets, hidden_layer_size = [100,100], alpha = 1e-02, batch = 32, n_iter = 100, random_state = 123):
    
    np.random.seed(random_state)
    
    input_size = inputs.shape[1]
    output_size = targets.shape[1]
    layer_sizes = [input_size] + hidden_layer_size + [output_size]
    
    weights, biases = initialize_weights(layer_sizes)
    
    error_l = []

    for j in range(n_iter):
        for i in range(0, len(inputs), batch):
            
            # BATCH RANGE 
            
            batch_start = i
            batch_end = i + batch
            
            input_batch = inputs[batch_start:batch_end]
            target_batch = targets[batch_start:batch_end]
            
            # FOWARD-PROPAGATION
            
            activations = forward_propagation(input_batch, weights, biases)
            a_output = activations[-1]

            # CROSS ENTROPY LOSS

            loss = cross_entropy_loss(a_output, target_batch)
            error_l.append(loss)

            # BACK-PROPAGATION 
            delta_output = a_output - target_batch
            deltas = [delta_output]
            for i in range(len(weights) - 1, 0, -1):
                delta = np.dot(deltas[-1], weights[i].T) * (activations[i] * (1 - activations[i]))
                deltas.append(delta)

            deltas.reverse()

            # WEIGHT UPDATE 
            for i in range(len(weights)):
                weights[i] -= alpha * np.dot(activations[i].T, deltas[i]) / batch
                biases[i] -= alpha * np.sum(deltas[i], axis=0) / batch
     
    return error_l, weights, biases

error, weights, biases = MLP_Classifier_Train(X_train, 
                                              y_train)

#%%

def MLP_Classifier_Test(inputs, weights, biases):
    
    predictions_ = []
    
    for i in range(len(inputs)):

        # FOWARD-PROPAGATION

        input = inputs[i:i+1].reshape((1,inputs.shape[1]))
        
        activations = forward_propagation(input, weights, biases)
        a_output = activations[-1]

        predictions_.append(np.argmax(a_output))
        
    return predictions_

predictions = MLP_Classifier_Test(X_test,
                                  weights,
                                  biases)

y_test = encoder.inverse_transform(y_test)

print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))
# %%

from matplotlib.animation import FuncAnimation

# Assuming you have X_test, y_test, and predictions
X_test = X_test.reshape((14000, 28, 28))

fig, ax = plt.subplots()
ax.set_title("Truth value: - Prediction: ")

def animate(i):
    ax.clear()
    ax.imshow(X_test[i])
    ax.set_title(f"Prediction: {predictions[i]} - Truth value: {int(y_test[i])}")
    ax.axis('off')

ani = FuncAnimation(fig, animate, frames=50, interval=1000)
ani.save('MLP_Classifier.gif', writer='ffmpeg')

plt.show()
#%%

plt.plot(error)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("MLP Classifier cross-entropy loss")
plt.savefig("MLP_Classifier_loss.png")
plt.show()
# %%
