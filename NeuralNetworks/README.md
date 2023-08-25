# Neural Networks

This subdirectory contains implementations and explanations of different neural network algorithms.

## 1. Perceptron

The Perceptron is a fundamental neural network model for binary classification. It's based on the concept of weighted inputs and a threshold activation function.

# $n = W \cdot p + b$

# $hardlim(n) = \begin{cases} 0 & \text{if } n < 0 \\ 1 & \text{if } n \geq 0 \end{cases}$


# $a = hardlim(n)$

# $W^{new} = W^{old} + e \cdot p^T$
# $b^{new} = b^{old} + e$
# where $e = t - a$

## 2. ADALINE (Adaptive Linear Neuron)

ADALINE is an improvement over the Perceptron, utilizing a continuous activation function and an adaptive weight adjustment mechanism.

Equation for ADALINE's output: 

$y = \sum_{i=1}^{n} w_i x_i + b$

## 3. MLP Regressor (Multi-Layer Perceptron Regressor)

The MLP Regressor is a type of neural network used for regression tasks. It consists of multiple layers of neurons, including input, hidden, and output layers.

## 4. MLP Classifier (Multi-Layer Perceptron Classifier)

The MLP Classifier is used for multiclass classification. It extends the MLP Regressor by using appropriate activation functions and output encoding.

Each neural network implementation in this subdirectory comes with detailed explanations and code samples.

For a detailed exploration of each algorithm, refer to the corresponding folders in this directory.

## License

This portion of the project is licensed under the [MIT License](../LICENSE).
