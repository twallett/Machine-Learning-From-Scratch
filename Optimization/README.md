# Optimization

This subdirectory includes implementations and explanations of various optimization techniques commonly used in machine learning.

## 1. Steepest Descent

Steepest Descent is an iterative optimization method used to find the minimum of a function. It involves moving in the direction of the negative gradient.

# $x_{k+1} = x_k - \alpha \nabla f(x_k)$

* $x_k$: The initial condition.
* $\alpha$: The learning rate.
* $\nabla f(x_k)$: The gradient.

## 2. Linear Minimization

Linear Minimization is an optimization technique that involves finding the minimum of a linear objective function subject to linear constraints.

# $\alpha_k = - \frac{\nabla f(x_k)^T \cdot p_k} {p_k^T \cdot  H_f(x_k) \cdot p_k} $

# $x_{k+1} = x_k - \alpha \nabla f(x_k)$

* $\alpha$: The learning rate.
* $x_k$: The initial condition.
* $p_k$: The search direction, or $-\nabla f(x_k)$
* $\nabla f(x_k)$: The gradient.
* $H_f(x_k)$: The Hessian.

## 3. Newton's Method

Newton's Method is an iterative optimization algorithm that uses second-order derivatives to find the minimum of a function more efficiently than gradient descent.

# $x_{k+1} = x_k - (H_f(x_k)^{-1} \cdot \nabla f(x_k))$

* $x_k$: The initial condition.
* $\alpha$: The learning rate.
* $\nabla f(x_k)$: The gradient.
* $H_f(x_k)$: The Hessian.

## 4. Conjugate Gradient Method

The Conjugate Gradient method is used to solve unconstrained optimization problems. It's particularly effective for large-scale optimization tasks.

# $x_{k+1} = x_k + \alpha_k p_k$

* $x_k$: The initial condition.
* $\alpha$: The learning rate.
* $p_k$: The search direction, or $-\nabla f(x_k)$

Each optimization technique in this subdirectory is explained in detail along with code examples.

For a deeper understanding of each method, refer to the corresponding folders in this directory.

## License

This portion of the project is licensed under the [MIT License](../LICENSE).
