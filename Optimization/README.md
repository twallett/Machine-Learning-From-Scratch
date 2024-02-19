# Optimization

This subdirectory includes implementations and explanations of various optimization techniques commonly used in machine learning.

Each optimization technique in this subdirectory is explained in detail along with code examples.

For a deeper understanding of each method, refer to the corresponding folders in this directory.

## 1. Stochastic Gradient Descent (SGD)

Stochastic Gradient Descent is an iterative optimization method used to find the minimum of a function. It involves moving in the direction of the negative gradient.

### $\theta_{k+1} = \theta_{k} - \alpha \nabla f({\theta_{k}})$

* $\theta_{k}$: The initial condition.
* $\alpha$: The learning rate.
* $\\nabla f({\theta_{k}})$: The gradient.

<table>
  <tr>
    <td style="width: 50%;">
      <img src="https://github.com/twallett/Machine-Learning-From-Scratch/blob/main/Optimization/1_SGD/SGD_contour.gif" alt="First GIF" width="100%">
    </td>
    <td style="width: 50%;">
      <img src="https://github.com/twallett/Machine-Learning-From-Scratch/blob/main/Optimization/1_SGD/SGD_surface.gif" alt="Second GIF" width="100%">
    </td>
  </tr>
</table>

## 2. Linear Minimization

Linear Minimization is an optimization technique that involves finding the minimum of a linear objective function subject to linear constraints.

### $\alpha_k = - \frac{\nabla f(\theta_{k})^T \cdot p_k} {p_k^T \cdot  H_f(\theta_{k}) \cdot p_k} $

### $\theta_{k+1} = \theta_{k} - \alpha \nabla f(\theta_{k})$

* $\alpha$: The learning rate.
* $\theta_{k}$: The initial condition.
* $p_k$: The search direction, or $-\nabla f(\theta_{k})$
* $\nabla f(\theta_{k})$: The gradient.
* $H_f(\theta_{k})$: The Hessian.

<table>
  <tr>
    <td style="width: 50%;">
      <img src="https://github.com/Twallett/Machine-Learning/blob/main/Optimization/2_Linear_minimization/Linear_minimization_contour.gif" alt="First GIF" width="100%">
    </td>
    <td style="width: 50%;">
      <img src="https://github.com/Twallett/Machine-Learning/blob/main/Optimization/2_Linear_minimization/Linear_minimization_surface.gif" alt="Second GIF" width="100%">
    </td>
  </tr>
</table>

## 3. Newton's Method

Newton's Method is an iterative optimization algorithm that uses second-order derivatives to find the minimum of a function more efficiently than gradient descent.

### $\theta_{k+1} = \theta_{k} - (H_f(\theta_{k})^{-1} \cdot \nabla f(\theta_{k}))$

* $\theta_{k}$: The initial condition.
* $\alpha$: The learning rate.
* $\nabla f(\theta_{k})$: The gradient.
* $H_f(\theta_{k})$: The Hessian.

<table>
  <tr>
    <td style="width: 50%;">
      <img src="https://github.com/Twallett/Machine-Learning/blob/main/Optimization/3_Newtons_method/Newtons_method_contour.gif" alt="First GIF" width="100%">
    </td>
    <td style="width: 50%;">
      <img src="https://github.com/Twallett/Machine-Learning/blob/main/Optimization/3_Newtons_method/Newtons_method_surface.gif" alt="Second GIF" width="100%">
    </td>
  </tr>
</table>

## 4. Conjugate Gradient Method

The Conjugate Gradient method is used to solve unconstrained optimization problems. It's particularly effective for large-scale optimization tasks.

### $\alpha_k = - \frac{\nabla f(\theta_{k})^T \cdot p_k} {p_k^T \cdot  H_f(\theta_{k}) \cdot p_k} $

### $\theta_{k+1} = \theta_{k} + \alpha_k p_k$

### $\beta_k = \frac{\nabla f(\theta_{k})^T \cdot \nabla f(\theta_{k})}{\nabla f(\theta_{k-1})^T \cdot \nabla f(\theta_{k-1})}$

### $p_k = -\nabla f(\theta_{k}) + \beta_k \cdot p_{k-1}$

* $\theta_{k}$: The initial condition.
* $\alpha$: The learning rate.
* $p_k$: The search direction, or $-\nabla f(\theta_{k})$
* $H_f(\theta_{k})$: The Hessian.
* $\beta_k$: The Fletcher and Reeves beta.

<table>
  <tr>
    <td style="width: 50%;">
      <img src="https://github.com/Twallett/Machine-Learning/blob/main/Optimization/4_Conjugate_method/Conjugate_method_contour.gif" alt="First GIF" width="100%">
    </td>
    <td style="width: 50%;">
      <img src="https://github.com/Twallett/Machine-Learning/blob/main/Optimization/4_Conjugate_method/Conjugate_method_surface.gif" alt="Second GIF" width="100%">
    </td>
  </tr>
</table>

## References 

Oklahoma State University–Stillwater. (n.d.). https://hagan.okstate.edu/NNDesign.pdf 
