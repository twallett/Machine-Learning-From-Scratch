# %%
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------
p = np.array([[1, 1, 2, 2, -1, -2, -1, -2],
              [1, 2, -1, 0, 2, 1, -1, -2]])

t = np.array([[-1, -1, -1, -1, 1, 1, 1, 1],
              [-1, -1, 1, 1, -1, -1, 1, 1]])
# -------------------------------------------------
# Plot the patterns and the targets

plt.scatter(p[0, :2], p[1, :2], marker='s')
plt.scatter(p[0, 2:4], p[1, 2:4], marker='o')
plt.scatter(p[0, 4:6], p[1, 4:6], marker='x')
plt.scatter(p[0, 6:8], p[1, 6:8], marker='P')
plt.ylabel('X2')
plt.xlabel('X1')
plt.title('X2 vs. X1')
plt.show()

#
# %%
# -------------------------------------------------
# Adaline network and use LMS

# wnew =  wold + 2*alpha*error*search direction

def adaline(inputs, labels, alpha, n_iter):
    inputs = np.repeat(inputs, n_iter, axis=1)
    labels = np.repeat(labels, n_iter, axis=1)
    w_init = np.identity(2)
    b_init = np.array([1, 1]).reshape((2, 1))
    error_l = []
    for i in range(1, inputs.shape[1] + 1):
        error = labels[:, i - 1:i] - (w_init @ inputs[:, i - 1:i] + b_init)
        w_init = w_init + 2 * alpha * error @ inputs[:, i - 1:i].T
        b_init = b_init + 2 * alpha * error
        error_l.append(error)
        if error[0] == 0 and error[1] == 0:
            break
        else:
            continue
    return w_init, b_init, error_l


w, b, e_list = adaline(p, t, 0.3, 300)

# -------------------------------------------------
# Calculate the sum square error and plot it

e_list_sq = np.array(e_list) ** 2

plt.plot(e_list_sq[:, 0])
plt.plot(e_list_sq[:, 1])
plt.legend(['X1', 'X2'])
plt.ylabel("Sum of squared errors")
plt.xlabel("Iterations")
plt.title("Sum of squared errors plot")
plt.show()

# %%
# -------------------------------------------------
# Plot the Decision Boundaries and the patterns.

fig, ax = plt.subplots()
plt.scatter(p[0, :2], p[1, :2], marker='s')
plt.scatter(p[0, 2:4], p[1, 2:4], marker='o')
plt.scatter(p[0, 4:6], p[1, 4:6], marker='x')
plt.scatter(p[0, 6:8], p[1, 6:8], marker='P')
plt.ylabel('X2')
plt.xlabel('X1')
plt.title('X2 vs. X1')

w_1, w_2 = w[0], w[1]
b_1, b_2 = b[0][0], b[1][0]

db1_slope = (-w_1[0]) / (w_1[1])
db2_slope = (-w_2[0]) / (w_2[1])

db1_int = (-b_1 / w_1[1])
db2_int = (-b_2 / w_2[1])

plt.quiver(0, 0, w_1[0], w_1[1], units='xy', scale=1)
ax.axline((0, db1_int), slope=db1_slope)

plt.quiver(0, 0, w_2[0], w_2[1], units='xy', scale=1)
ax.axline((0, db2_int), slope=db2_slope)

plt.show()


# %%
# -------------------------------------------------
# Compare the results of part iv. with decision boundary that you will get with perceprtron learning rule. Explain the differences?

def perceptron(inputs, labels, alpha, n_iter):
    inputs = np.repeat(inputs, n_iter, axis=1)
    labels = np.repeat(labels, n_iter, axis=1)
    w_init = np.identity(2)
    b_init = np.array([1, 1]).reshape((2, 1))
    error_l = []
    for i in range(1, inputs.shape[1] + 1):
        error = labels[:, i - 1:i] - (w_init @ inputs[:, i - 1:i] + b_init)
        w_init = w_init + 2 * alpha * error @ inputs[:, i - 1:i].T
        b_init = b_init + 2 * alpha * error
        error_l.append(error)
        if error[0] == 0 and error[1] == 0:
            break
        else:
            continue
    return w_init, b_init, error_l
