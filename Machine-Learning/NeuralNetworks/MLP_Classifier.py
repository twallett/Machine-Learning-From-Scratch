#%%
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

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

#%%

# Activation function in last layer
def softmax(n):
    n = n.astype(float)
    result = np.empty_like(n)
    for i in range(n.shape[0]):
        result[i,0] = np.exp(n[i,0]) / np.sum(np.exp(n[:,0]))
    return result

# Activation function in hidden layer
def logsig(n):
    n = n.astype(float)
    result = np.empty_like(n)
    for i in range(n.shape[0]):
        result[i,0] = 1 / (1 + np.exp(-n[i,0]))
    return result

# Loss function
def cross_entropy_loss(prob, actual):
    return -np.sum(actual.T * np.log(prob))

def MLP_Classifier_Train(inputs, targets, s_size, alpha, n_iter, random_state = 123):
    np.random.seed(random_state)
    
    w_init_1 = np.random.rand(s_size, inputs.shape[1]) * 0.01
    b_init_1 = np.random.rand(s_size, 1) * 0.01

    w_init_2 = np.random.rand(targets.shape[1], s_size)
    b_init_2 = np.random.rand(targets.shape[1], 1)

    error_l = []

    for epoch in range(n_iter):
        for i in range(len(inputs)):

            # FOWARD-PROPAGATION

            a_init = inputs[i:i+1].reshape((inputs.shape[1],1))
            
            n1 = (w_init_1 @ a_init) + b_init_1

            a1 = logsig(n1)
            
            n2 = (w_init_2 @ a1) + b_init_2

            a2 = softmax(n2)

            target = targets[i:i+1].reshape(targets.shape[1],1)

            # CROSS ENTROPY LOSS

            error = cross_entropy_loss(a2, target)
            error_l.append(error)

            # BACK-PROPAGATION 

            s2_b = (a2 - target)
            s2 = s2_b @ a1.T

            f_1 = np.diagflat(a1) @ np.diagflat((1-a1))
            
            s1 =  f_1 @ (w_init_2.T @ s2_b)

            # WEIGHT UPDATE 

            w_init_2 -= (alpha * s2)
            b_init_2 -= (alpha * s2_b)

            w_init_1 -= (alpha * s1 @ a_init.T)
            b_init_1 -= (alpha * s1)

    return error_l, w_init_1, b_init_1, w_init_2, b_init_2

error, w1, b1, w2, b2 = MLP_Classifier_Train(X_train,
                                             y_train, 
                                             s_size=100,
                                             alpha = 1e-04,
                                             n_iter=5)

#%%

def MLP_Classifier_Test(inputs, weight1, bias1, weight2, bias2):
    
    predictions_ = []
    
    for i in range(len(inputs)):

        # FOWARD-PROPAGATION

        a_init = inputs[i:i+1].reshape((inputs.shape[1],1))
        
        n1 = (weight1 @ a_init) + bias1

        a1 = logsig(n1)
        
        n2 = (weight2 @ a1) + bias2

        a2 = softmax(n2) 
        
        predictions_.append(np.argmax(a2))
        
    return predictions_

predictions = MLP_Classifier_Test(X_test, 
                                  w1, 
                                  b1, 
                                  w2, 
                                  b2)

y_test = encoder.inverse_transform(y_test)

print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))

