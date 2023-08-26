# Neural Networks

This subdirectory contains implementations and explanations of different neural network algorithms.

Each neural network implementation in this subdirectory comes with detailed explanations and code samples.

For a detailed exploration of each algorithm, refer to the corresponding folders in this directory.


## 1. Perceptron

The Perceptron is a fundamental neural network model for binary classification. It's based on the concept of weighted inputs and a threshold activation function.

# $n = W \cdot p + b$

# $a = hardlim(n)$

# $W^{new} = W^{old} + e \cdot p^T$
# $b^{new} = b^{old} + e$
# $where \ e = t - a$

<table>
  <tr>
    <td style="width: 50%;">
      <img src="https://github.com/Twallett/Machine-Learning/blob/main/NeuralNetworks/1_Perceptron/Perceptron_classification.gif" alt="First GIF" width="100%">
    </td>
    <td style="width: 50%;">
      <img src="https://github.com/Twallett/Machine-Learning/blob/main/NeuralNetworks/1_Perceptron/Perceptron_classification_XOR.gif" alt="Second GIF" width="100%">
    </td>
  </tr>
</table>
<table>
  <tr>
    <td style="width: 50%;">
      <img src="https://github.com/Twallett/Machine-Learning/blob/main/NeuralNetworks/1_Perceptron/Perceptron_classification.gif" alt="First GIF" width="100%">
    </td>
    <td style="width: 50%;">
      <img src="https://github.com/Twallett/Machine-Learning/blob/main/NeuralNetworks/1_Perceptron/Perceptron_classification_XOR.gif" alt="Second GIF" width="100%">
    </td>
  </tr>
</table>

## 2. ADALINE (Adaptive Linear Neuron)

ADALINE is an improvement over the Perceptron, utilizing a continuous activation function and an adaptive weight adjustment mechanism.

# $a = purelin(W \cdot p + b)$

# $W_{k+1} = W_{k} - 2 \alpha e_{k} \cdot p_{k}^T$
# $b_{k+1} = b_{k} - 2 \alpha e_{k}$

## 3. MLP Regressor (Multi-Layer Perceptron Regressor)

The MLP Regressor is a type of neural network used for regression tasks. It consists of multiple layers of neurons, including input, hidden, and output layers.

# $Forward propagation:$
# $a^{m+1} = f^{m+1}(W^{m+1} \cdot a^m + b^{m+1})\ for \ m = 0, 1, ..., M-1$

# $Backward propagation:$
# $s^{M} = F^{M} \cdot (n^{M}) \cdot e$
# $s^{m} = F^{m} \cdot (n^{m}) \cdot (W^{m+1^{T}}) \cdot s^{m+1}$

# $W_{k+1}^m = W_{k}^m - \alpha s^m \cdot (a^{{m-1}^T})$
# $b_{k+1}^m = b_{k}^m - \alpha s^m$

## 4. MLP Classifier (Multi-Layer Perceptron Classifier)

The MLP Classifier is used for multiclass classification. It extends the MLP Regressor by using appropriate activation functions and output encoding.

# $Forward propagation:$
# $a^{m+1} = f^{m+1}(W^{m+1} \cdot a^m + b^{m+1}) \ for \ m = 0, 1, ..., M-1$

## References 

Oklahoma State University–Stillwater. (n.d.). https://hagan.okstate.edu/NNDesign.pdf 

## License

This portion of the project is licensed under the [MIT License](../LICENSE).
