# Optimization

This subdirectory includes implementations and explanations of various optimization techniques commonly used in machine learning.

## 1. Steepest Descent

Steepest Descent is an iterative optimization method used to find the minimum of a function. It involves moving in the direction of the negative gradient.

Equation for Steepest Descent update: $x_{k+1} = x_k - \alpha \nabla f(x_k)$

## 2. Linear Minimization

Linear Minimization is an optimization technique that involves finding the minimum of a linear objective function subject to linear constraints.

Equation for Linear Minimization problem: $\text{minimize } c^T x \text{ subject to } Ax \leq b$

## 3. Newton's Method

Newton's Method is an iterative optimization algorithm that uses second-order derivatives to find the minimum of a function more efficiently than gradient descent.

Equation for Newton's Method update: $x_{k+1} = x_k - H_f(x_k)^{-1} \nabla f(x_k)$

## 4. Conjugate Gradient Method

The Conjugate Gradient method is used to solve unconstrained optimization problems. It's particularly effective for large-scale optimization tasks.

Equation for Conjugate Gradient update: $x_{k+1} = x_k + \alpha_k p_k$

Each optimization technique in this subdirectory is explained in detail along with code examples.

For a deeper understanding of each method, refer to the corresponding folders in this directory.

## License

This portion of the project is licensed under the [MIT License](../LICENSE).
