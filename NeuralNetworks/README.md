# Neural Networks

This subdirectory contains implementations and explanations of different neural network algorithms.

Each neural network implementation in this subdirectory comes with detailed explanations and code samples.

For a detailed exploration of each algorithm, refer to the corresponding folders in this directory.

## 1. Perceptron

The Perceptron is a fundamental neural network model for binary classification. It's based on the concept of weighted inputs and a threshold activation function.

# $Forwardpropagation:$
# $n = W \cdot p + b$
# $a = hardlim(n)$

# $Weight \ updates:$
# $W^{new} = W^{old} + e \cdot p^T$
# $b^{new} = b^{old} + e$
# $where \ e = t - a$

<table>
  <tr>
    <td style="width: 50%;">
      <img src="https://github.com/Twallett/Machine-Learning/blob/main/NeuralNetworks/1_Perceptron/Perceptron_classification.gif" alt="First GIF" width="100%">
    </td>
    <td style="width: 50%;">
      <img src="https://github.com/Twallett/Machine-Learning/blob/main/NeuralNetworks/1_Perceptron/Perceptron_sse.png" alt="Second GIF" width="100%">
    </td>
  </tr>
  <tr>
    <td style="width: 50%;">
      <img src="https://github.com/Twallett/Machine-Learning/blob/main/NeuralNetworks/1_Perceptron/Perceptron_classification_XOR.gif" alt="Third GIF" width="100%">
    </td>
    <td style="width: 50%;">
      <img src="https://github.com/Twallett/Machine-Learning/blob/main/NeuralNetworks/1_Perceptron/Perceptron_sse_XOR.png" alt="Fourth GIF" width="100%">
    </td>
  </tr>
</table>


## 2. ADALINE (Adaptive Linear Neuron)

ADALINE is an improvement over the Perceptron, utilizing a continuous activation function and an adaptive weight adjustment mechanism.

# $Forwardpropagation:$
# $a = purelin(W \cdot p + b)$

# $Weight \ updates:$
# $W_{k+1} = W_{k} - 2 \alpha e_{k} \cdot p_{k}^T$
# $b_{k+1} = b_{k} - 2 \alpha e_{k}$

<table>
  <tr>
    <td style="width: 50%;">
      <img src="https://github.com/Twallett/Machine-Learning/blob/main/NeuralNetworks/2_ADALINE/ADALINE_classification.gif" alt="First GIF" width="100%">
    </td>
    <td style="width: 50%;">
      <img src="https://github.com/Twallett/Machine-Learning/blob/main/NeuralNetworks/2_ADALINE/ADALINE_sse.png" alt="Second GIF" width="100%">
    </td>
  </tr>
</table>

## 3. MLP Regressor (Multi-Layer Perceptron Regressor)

The MLP Regressor is a type of neural network used for regression tasks. It consists of multiple layers of neurons, including input, hidden, and output layers.

# $Forwardpropagation:$
# $a^0 = p$
# $a^{m+1} = f^{m+1}(W^{m+1} \cdot a^m + b^{m+1})\ for \ m = 0, 1, ..., M-1$
# $a = a^M$

# $Backpropagation:$
# $s^{M} = F^{M} \cdot (n^{M}) \cdot e$
# $s^{m} = F^{m} \cdot (n^{m}) \cdot (W^{m+1^{T}}) \cdot s^{m+1} \ for \ m = M-1, ..., 2, 1$

# $Weight \ updates:$
# $W_{k+1}^m = W_{k}^m - \alpha s^m \cdot (a^{{m-1}^T})$
# $b_{k+1}^m = b_{k}^m - \alpha s^m$

<table>
  <tr>
    <td style="width: 50%;">
      <img src="https://github.com/Twallett/Machine-Learning/blob/main/NeuralNetworks/3_MLP_Regressor/MLP_Regressor_target.png" alt="First GIF" width="100%">
    </td>
    <td style="width: 50%;">
      <img src="https://github.com/Twallett/Machine-Learning/blob/main/NeuralNetworks/3_MLP_Regressor/MLP_Regressor_target_predictions.png" alt="Second GIF" width="100%">
    </td>
  </tr>
  <tr>
    <td style="width: 50%;">
      <img src="https://github.com/Twallett/Machine-Learning/blob/main/NeuralNetworks/3_MLP_Regressor/MLP_Regressor_sse.png" alt="Third GIF" width="100%">
    </td>
    <td style="width: 50%;">
      <img src="https://github.com/Twallett/Machine-Learning/blob/main/NeuralNetworks/3_MLP_Regressor/MLP_Regressor_target_scatter.png" alt="Fourth GIF" width="100%">
    </td>
  </tr>
</table>

## 4. MLP Classifier (Multi-Layer Perceptron Classifier)

The MLP Classifier is used for multiclass classification. It extends the MLP Regressor by using appropriate activation functions and output encoding.

# $Forwardpropagation:$
# $a^0 = p$
# $a^{m+1} = f^{m+1}(W^{m+1} \cdot a^m + b^{m+1}) \ for \ m = 0, 1, ..., M-2$
# $a^M = softmax(W^{m+1} \cdot a^{M-1} + b^{m+1})\ for \ m = M-1$
# $a = a^M$

# $Backpropagation:$
# $s^{M} = a - t$
# $s^{m} = F^{m} \cdot (n^{m}) \cdot (W^{m+1^{T}}) \cdot s^{m+1} \ for \ m = M-1, ..., 2, 1$

# $Weight \ updates:$
# $W_{k+1}^m = W_{k}^m - \alpha s^m \cdot (a^{{m-1}^T})$
# $b_{k+1}^m = b_{k}^m - \alpha s^m$

<table>
  <tr>
    <td style="width: 50%;">
      <img src="https://github.com/Twallett/Machine-Learning/blob/main/NeuralNetworks/4_MLP_Classifier/MLP_Classifier.gif" alt="First GIF" width="100%">
    </td>
    <td style="width: 50%;">
      <img src="https://github.com/Twallett/Machine-Learning/blob/main/NeuralNetworks/4_MLP_Classifier/MLP_Classifier_loss.png" alt="Second GIF" width="100%">
    </td>
  </tr>
</table>

## References 

Oklahoma State University–Stillwater. (n.d.). https://hagan.okstate.edu/NNDesign.pdf 
