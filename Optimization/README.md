# Optimization

This subdirectory includes implementations and explanations of various optimization techniques commonly used in machine learning.

## 1. Steepest Descent

Steepest Descent is an iterative optimization method used to find the minimum of a function. It involves moving in the direction of the negative gradient.

# $\theta_{k+1} = \theta_{k} - \alpha \nabla f({\theta})$

* $\theta_{k}$: The initial condition.
* $\alpha$: The learning rate.
* $\\nabla f({\theta})$: The gradient.

<table>
  <tr>
    <td style="width: 50%;">
      <img src="https://github.com/Twallett/Machine-Learning/blob/main/Optimization/1_Steepest_descent/Steepest_descent_contour.gif" alt="First GIF" width="100%">
    </td>
    <td style="width: 50%;">
      <img src="https://github.com/Twallett/Machine-Learning/blob/main/Optimization/1_Steepest_descent/Steepest_descent_surface.gif" alt="Second GIF" width="100%">
    </td>
  </tr>
</table>

## 2. Linear Minimization

Linear Minimization is an optimization technique that involves finding the minimum of a linear objective function subject to linear constraints.

# $\alpha_k = - \frac{\nabla f(x_k)^T \cdot p_k} {p_k^T \cdot  H_f(x_k) \cdot p_k} $

# $x_{k+1} = x_k - \alpha \nabla f(x_k)$

* $\alpha$: The learning rate.
* $x_k$: The initial condition.
* $p_k$: The search direction, or $-\nabla f(x_k)$
* $\nabla f(x_k)$: The gradient.
* $H_f(x_k)$: The Hessian.

<table>
  <tr>
    <td style="width: 50%;">
      <img src="https://github.com/Twallett/Machine-Learning/blob/main/Optimization/1_Steepest_descent/Steepest_descent_contour.gif" alt="First GIF" width="100%">
    </td>
    <td style="width: 50%;">
      <img src="https://github.com/Twallett/Machine-Learning/blob/main/Optimization/1_Steepest_descent/Steepest_descent_surface.gif" alt="Second GIF" width="100%">
    </td>
  </tr>
</table>

## 3. Newton's Method

Newton's Method is an iterative optimization algorithm that uses second-order derivatives to find the minimum of a function more efficiently than gradient descent.

# $x_{k+1} = x_k - (H_f(x_k)^{-1} \cdot \nabla f(x_k))$

* $x_k$: The initial condition.
* $\alpha$: The learning rate.
* $\nabla f(x_k)$: The gradient.
* $H_f(x_k)$: The Hessian.

## 4. Conjugate Gradient Method

The Conjugate Gradient method is used to solve unconstrained optimization problems. It's particularly effective for large-scale optimization tasks.

# $\alpha_k = - \frac{\nabla f(x_k)^T \cdot p_k} {p_k^T \cdot  H_f(x_k) \cdot p_k} $

# $x_{k+1} = x_k + \alpha_k p_k$

# $\beta_k = \frac{\nabla f(x_k)^T \cdot \nabla f(x_k)}{\nabla f(x_{k-1})^T \cdot \nabla f(x_{k-1})}$

# $p_k = -\nabla f(x_k) + \beta_k \cdot p_{k-1}$

* $x_k$: The initial condition.
* $\alpha$: The learning rate.
* $p_k$: The search direction, or $-\nabla f(x_k)$

Each optimization technique in this subdirectory is explained in detail along with code examples.

For a deeper understanding of each method, refer to the corresponding folders in this directory.

## License

This portion of the project is licensed under the [MIT License](../LICENSE).
