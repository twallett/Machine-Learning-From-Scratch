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
    n = n.astype(float)
    result = np.empty_like(n)
    for i in range(n.shape[0]):
        result[i,0] = 1 / (1 + np.exp(-n[i,0]))
    return result

def MLP_Regressor(inputs_, targets_, alpha, s_size, n_iter):
        
    w_init_1 = np.random.rand(s_size, inputs.shape[1]) * 0.01
    b_init_1 = np.random.rand(s_size, 1) * 0.01

    w_init_2 = np.random.rand(targets.shape[1], s_size)* 0.01
    b_init_2 = np.random.rand(targets.shape[1], 1)* 0.01

    error_l = [] 
    output_l = []

    for j in range(n_iter):
        for i in range(len(inputs_)):
            
            # FEEDFOWARD PART
        
            a_init = inputs_[i:i+1].T

            n1 = (w_init_1 @ a_init) + b_init_1

            a1 = logsig(n1)

            a2 = (w_init_2 @ a1) + b_init_2

            target = targets_[i:i+1].T

            error = ((target - a2)).item()

            #BACKWARDPROPAGATION
            
            f_1_1 = (np.ones((s_size,1)) - a1)
            f_1 = (np.matrix(np.diag([f_1_1[i,0] for i in range(s_size)]))) @ (np.matrix(np.diag([a1[i,0] for i in range(s_size)])))

            s2 = -2 * error
            s1 = f_1 @ (w_init_2.T * s2)
            
            error_l.append(error ** 2)
            output_l.append(a2)
                
            w_init_2 -= (alpha * s2 * a1.T)
            b_init_2 -= (alpha * s2)

            w_init_1 -= (alpha * s1 @ a_init.T)
            b_init_1 -= (alpha * s1)
            
    return error_l, output_l 
        
error, output = MLP_Regressor(inputs,
                              targets,
                              alpha= 1e-2,
                              s_size= 2, 
                              n_iter=100)


