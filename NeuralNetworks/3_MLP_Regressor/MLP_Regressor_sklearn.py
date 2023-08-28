#%%

import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.neural_network import MLPRegressor

def f(x):
    return 1 + np.sin((math.pi/4) * x) 

function_latex = "$f(x) = 1 + sin(\\frac{\pi}{4}x)$"
inputs = np.linspace(-2,2).reshape(-1,1)
targets = f(np.linspace(-2,2)).reshape(-1,1)

model = MLPRegressor(hidden_layer_sizes=(600,600,600),
                     max_iter=10000,
                     random_state=123).fit(inputs, targets)

print(f"The R^2 score: {model.score(inputs, targets)}")

plt.plot(model.loss_curve_)
plt.ylabel('SSE')
plt.xlabel('Iterations')
plt.title("MLP Regressor sklearn sum of squared errors")
plt.show()

plt.plot(model.predict(inputs))
plt.plot(targets)
plt.ylabel('$Y$')
plt.xlabel('$X$')
plt.title(f"MLP Regressor sklearn of {function_latex}")
plt.show()

# %%
