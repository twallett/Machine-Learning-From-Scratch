# Neural Networks

This subdirectory contains implementations and explanations of different neural network algorithms.

## 1. Perceptron

The Perceptron is a fundamental neural network model for binary classification. It's based on the concept of weighted inputs and a threshold activation function.

Equation for a Perceptron's output: 

$y = \begin{cases} 1, & \text{if } \sum_{i=1}^{n} w_i x_i + b > \text{threshold} \\ 0, & \text{otherwise} \end{cases}$

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
