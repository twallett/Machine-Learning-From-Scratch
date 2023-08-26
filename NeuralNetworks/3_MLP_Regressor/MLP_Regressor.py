#%%
import numpy as np
import matplotlib.pyplot as plt
import math

def f(x):
    return 1 + np.sin((math.pi/4) * x) 

function_latex = "$f(x) = 1 + sin(\\frac{\pi}{4}x)$"
inputs = np.linspace(-2,2).reshape(-1,1)
targets = f(np.linspace(-2,2)).reshape(-1,1)
#%%

def logsig(n):
    return 1 / (1 + np.exp(-n))

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
    a_output = z_output
    activations.append(a_output)
    return activations

def MLP_Regressor_Train(inputs_, targets_, hidden_sizes = [100,100], alpha = 1e-04, batch = 32, n_iter = 1000):
    
    input_size = inputs_.shape[1]
    output_size = targets_.shape[1]
    layers_sizes = [input_size] + hidden_sizes + [output_size]
    
    weights, biases = initialize_weights(layers_sizes)

    error_l = [] 
    output_l = []

    for j in range(n_iter):
        for i in range(0, len(inputs_), batch):
            
            # BATCH RANGE 
            
            batch_start = i
            batch_end = i + batch
            
            input_batch = inputs_[batch_start:batch_end]
            target_batch = targets_[batch_start:batch_end]
            
            # FORWARD PROPAGATION 

            activations = forward_propagation(input_batch, weights, biases)
            a_output = activations[-1]
        
            # LOSS CALCULATION
            error = np.mean((target_batch - a_output) ** 2)
            
            error_l.append(error)
            output_l.append(a_output)

            # BACK PROPAGATION AND WEIGHT UPDATES
            
            gradients = [-2 * (target_batch - a_output) / batch]
            for i in range(len(weights) - 2, -1, -1):
                derivative_n = gradients[-1].dot(weights[i+1].T) * activations[i+1] * (1 - activations[i+1])
                derivative_weight = np.dot(activations[i].T, derivative_n)
                derivative_bias = np.sum(derivative_n, axis= 0)
                gradients.append(derivative_n)
                weights[i] -= alpha * derivative_weight
                biases[i] -= alpha * derivative_bias
            
    return error_l, output_l, weights, biases

error, output, weights, biases = MLP_Regressor_Train(inputs,
                                                    targets,
                                                    alpha=0.05)

#%%

def MLP_Regressor_Test(inputs, weights, biases):
    
    predictions_ = []
    
    for i in range(len(inputs)):

        # FOWARD-PROPAGATION

        input = inputs[i:i+1].reshape((1,inputs.shape[1]))
        
        activations = forward_propagation(input, weights, biases)
        a_output = activations[-1]

        predictions_.append(a_output)
        
    return predictions_

predictions = MLP_Regressor_Test(inputs,
                                weights,
                                biases)

# %%

plt.plot(targets)
plt.ylabel('$Y$')
plt.xlabel('$X$')
plt.title(f"MLP Regressor of underlying function {function_latex}")
plt.tight_layout()
plt.savefig("MLP_Regressor_target.png")
plt.show()
# %%

plt.plot(targets, label = 'Target')
plt.plot(np.concatenate(predictions), label = 'Predictions')
plt.legend()
plt.ylabel('$Y$')
plt.xlabel('$X$')
plt.title(f"MLP Regressor of {function_latex}")
plt.savefig("MLP_Regressor_target_predictions.png")
plt.show()

# %%

plt.plot(error)
plt.ylabel('SSE')
plt.xlabel('Iterations')
plt.title("MLP Regressor sum of squared errors")
plt.savefig("MLP_Regressor_sse.png")
plt.show()

# %%

x = np.linspace(0, 2, 100)
y = x

plt.plot(x, y, label='y = x', color='black')
plt.scatter(predictions, targets)
plt.ylabel('$y$')
plt.xlabel('$a^M$')
plt.title("MLP Regressor scatterplot of targets vs. predictions")
plt.savefig("MLP_Regressor_target_scatter.png")
plt.show()
# %%
