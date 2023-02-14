# %%
# Question 9:
# i. Write a script to initialize and train a network to solve this ”practical” problem. Hint: Implement the learning rule
import numpy as np


def transfer_function(n):
    if n < 0:
        return 0
    else:
        return 1


def perceptron(inputs, labels, n_iter):
    inputs = np.array(inputs * n_iter)
    labels = labels * n_iter
    w_init = np.random.rand(1, 2).round(1)
    b_init = 0
    error_list = []
    for i in range(len(inputs)):
        n = np.dot(w_init, inputs[i]) + b_init  # n = W*p + b
        a = transfer_function(n)  # a = F(n)
        error = labels[i] - a
        error_list.append(error)
        if error == 1:
            w_init = w_init + inputs[i]
            b_init = b_init + error
        elif error == -1:
            w_init = w_init - inputs[i]
            b_init = b_init + error
        elif error == 0:
            w_init = w_init
            b_init = b_init
    return w_init, b_init, error_list


# %%
# ii. Test the resulting weight and bias values against the input vectors.

inputs = [[1, 4], [1, 5], [2, 4], [2, 5], [3, 1], [3, 2], [4, 1], [4, 2]]
labels = [0, 0, 0, 0, 1, 1, 1, 1]

weight, bias, error_list = perceptron(inputs, labels, 2)

print(f"Resulting weight vector: {weight}", '\n')
print(f"Resulting bias: {bias}", '\n')
print(f"Resulting errors: {error_list}", '\n')

# %%
# iii. Plot error vs epochs and plot the trained network its decision boundary with weight vector and its direction.

import matplotlib.pyplot as plt

# Perceptron decision boundary
p1 = np.array(inputs)[:, 0]
p2 = np.array(inputs)[:, 1]

p1_w = float(weight[:, 0])
p2_w = float(weight[:, 1])

db_slope = (-p1_w) / p2_w
db_intercept = (-bias) / p2_w

db_1 = db_slope + db_intercept

fig, ax = plt.subplots()
plt.scatter(p1, p2, c=labels)
plt.quiver(1, db_1, p1_w, p2_w, units='xy', scale=1)
ax.axline((0, db_intercept), slope=db_slope)
plt.title("Perceptron decision boundary")
plt.xlabel("P1")
plt.ylabel("P2")
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()

# Error vs. epochs
plt.plot(error_list)
plt.title("Error vs. epochs")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

# %%
# iv. Alter the input vectors to ensure that the decision boundary of any solution will not intersect one of the original input vectors (i.e., to ensure only robust solutions are found). Then retrain the network.

inputs = [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [2, 3], [2, 4]]
labels = [0, 0, 0, 0, 1, 1, 1, 1]

weight, bias, error_list = perceptron(inputs, labels, 100)

print(f"Resulting weight vector: {weight}", '\n')
print(f"Resulting bias: {bias}", '\n')
print(f"Resulting errors: {error_list}", '\n')

# Perceptron decision boundary of new inputs and labels
p1 = np.array(inputs)[:, 0]
p2 = np.array(inputs)[:, 1]

p1_w = float(weight[:, 0])
p2_w = float(weight[:, 1])

db_slope = (-p1_w) / p2_w
db_intercept = (-bias) / p2_w

db_1 = db_slope + db_intercept

fig, ax = plt.subplots()
plt.scatter(p1, p2, c=labels)
plt.quiver(1, db_1, p1_w, p2_w, units='xy', scale=1)
ax.axline((0, db_intercept), slope=db_slope)
plt.title("Perceptron decision boundary")
plt.xlabel("P1")
plt.ylabel("P2")
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()

# Error vs. epochs of new inputs and labels
plt.plot(error_list)
plt.title("Error vs. epochs")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

# %%
# Extra: unsolvable problem
new_inputs = [[1, 1], [1, 4], [4, 1], [4, 4]]
new_labels = [1, 0, 0, 1]

new_weight, new_bias, new_error_list = perceptron(new_inputs, new_labels, 100)

plt.plot(new_error_list)
plt.title("Error vs. epochs")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

p1 = np.array(new_inputs)[:, 0]
p2 = np.array(new_inputs)[:, 1]
origin = [0, 0]

p1_w = float(new_weight[:, 0])
p2_w = float(new_weight[:, 1])

db_slope = (-p1_w) / p2_w
db_intercept = (-new_bias) / p2_w

fig, ax = plt.subplots()
plt.scatter(p1, p2, c=new_labels)
ax.axline((0, db_intercept), slope=db_slope)
plt.title("Perceptron decision boundary")
plt.xlabel("P1")
plt.ylabel("P2")
plt.show()

# %%
# Question 6: part v.

q6_v = [[-1, 0], [1, 2], [-1, 1], [0, 2]]
q6_v_labels = [1, 1, 0, 0]

q6_v_weight, q6_v_bias, q6_v_error_list = perceptron(q6_v, q6_v_labels, 15)

q6_v_p1 = np.array(q6_v)[:, 0]
q6_v_p2 = np.array(q6_v)[:, 1]
origin = [0, 0]

q6_v_p1_w = float(q6_v_weight[:, 0])
q6_v_p2_w = float(q6_v_weight[:, 1])

q6_v_db_slope = (-q6_v_p1_w) / q6_v_p2_w
q6_v_db_intercept = (-q6_v_bias) / q6_v_p2_w

fig, ax = plt.subplots()
plt.scatter(q6_v_p1, q6_v_p2, c=q6_v_labels)
ax.axline((0, q6_v_db_intercept), slope=q6_v_db_slope)
plt.show()

plt.plot(q6_v_error_list)
plt.show()

# %%
#rand
