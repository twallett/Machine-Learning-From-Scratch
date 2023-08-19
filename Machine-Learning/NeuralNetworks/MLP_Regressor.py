#%%

import numpy as np
import matplotlib.pyplot as plt
import math

def f(x):
    return 1 + np.sin((math.pi/4) * x)

inputs = np.linspace(-2,2).reshape(-1,1)
targets = f(np.linspace(-2,2)).reshape(-1,1)

#%%

def logsig(n):
    return 1 / (1 + np.exp(-n))

def MLP_Regressor(inputs_, targets_, alpha = 1e-04, s_size = 500, batch = 32, n_iter = 200):
        
    w_init_1 = np.random.rand(inputs.shape[1], s_size) 
    b_init_1 = np.random.rand(1, s_size) 

    w_init_2 = np.random.rand(s_size, 1) 
    b_init_2 = np.random.rand(1, 1) 

    error_l = [] 
    output_l = np.zeros((len(inputs_), 1)) 

    for j in range(n_iter):
        for i in range(0, len(inputs_), batch):
            
            # BATCH RANGE 
            
            batch_start = i
            batch_end = i + batch
            
            # FEEDFOWARD PART
        
            a_init = inputs_[batch_start:batch_end]

            n1 = np.dot(a_init, w_init_1) + b_init_1

            a1 = logsig(n1)

            a2 = np.dot(a1, w_init_2) + b_init_2

            target = targets_[batch_start:batch_end]

            error = (target - a2)

            #BACKWARDPROPAGATION
            
            s2_w = -2 * error / batch
            s2_b = np.sum(s2_w, axis= 0)
            
            s1_a = np.dot(s2_w, w_init_2.T)
            s1_w = s1_a * (a1 * (1 - a1))
            s1_b = np.sum(s1_w, axis= 0)
            
            w_init_2 -= (alpha * np.dot(a1.T, s2_w)) 
            b_init_2 -= (alpha * s2_b)

            w_init_1 -= (alpha * np.dot(a_init.T, s1_w))
            b_init_1 -= (alpha * s1_b)
            
            error_l.append(np.mean(error) ** 2)
            output_l[batch_start:batch_end] = a2
            
    return error_l, output_l 

error, output = MLP_Regressor(inputs,
                              targets, 
                              batch= 8,
                              n_iter=100000)


#%%

plt.plot(np.array(error).reshape(-1,1))
plt.show()

plt.plot(output)
plt.plot(targets)
plt.show()

# %%