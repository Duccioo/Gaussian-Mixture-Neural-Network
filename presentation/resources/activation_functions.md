# Activation Functions Formulas

This document presents the mathematical formulas for four common activation functions used in neural networks: ReLU, Sigmoid, Tanh, and Adaptive Sigmoid.

## 1. ReLU (Rectified Linear Unit)

The ReLU function is defined as:

$$f(x) = \max(0, x)$$

Or in piecewise notation:

$$
f(x) = \begin{cases}
    x & \text{if } x > 0 \\
    0 & \text{if } x \leq 0
\end{cases}
$$

Derivative:

$$
f'(x) = \begin{cases}
    1 & \text{if } x > 0 \\
    0 & \text{if } x \leq 0
\end{cases}
$$

## 2. Sigmoid

The Sigmoid function is defined as:

$$f(x) = \frac{1}{1 + e^{-x}}$$

Derivative:

$$f'(x) = f(x)(1 - f(x))$$

## 3. Tanh (Hyperbolic Tangent)

The Tanh function is defined as:

$$f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

Alternatively, it can be expressed in terms of the sigmoid function:

$$f(x) = 2\sigma(2x) - 1$$

where $$\sigma(x)$$ is the sigmoid function.

Derivative:

$$f'(x) = 1 - \tanh^2(x)$$

## 4. Adaptive Sigmoid

The Adaptive Sigmoid function introduces a learnable parameter $$\alpha$$ to adjust the steepness of the sigmoid curve:

$$f(x) = \frac{1}{1 + e^{-\alpha x}}$$

where $$\alpha$$ is a learnable parameter.

Derivative with respect to x:

$$\frac{\partial f}{\partial x} = \alpha f(x)(1 - f(x))$$

Derivative with respect to $$\alpha$$:

$$\frac{\partial f}{\partial \alpha} = xf(x)(1 - f(x))$$

These activation functions play crucial roles in neural networks by introducing non-linearity, allowing the network to learn complex patterns and relationships in the data.
