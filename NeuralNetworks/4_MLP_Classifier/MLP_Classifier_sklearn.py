#%%

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
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

model = MLPClassifier(hidden_layer_sizes=(300,300)).fit(X_train, y_train)

y_test = encoder.inverse_transform(y_test)
y_pred = model.predict(X_test)
y_pred = np.array([np.argmax(y_pred[i]) for i in range(len(y_pred))])

print(f"The accuracy score: {accuracy_score(y_test, y_pred)}")
print(f"The confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
# %%
