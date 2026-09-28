# Linear Regression from Scratch Using Gradient Descent

A simple implementation of **univariate linear regression using gradient descent**, applied to the California Housing dataset.

## Overview

This project implements linear regression from scratch without using a machine-learning library to train the model.

The model uses:

- **Median Income** as the input feature
- **Median House Value** as the target variable
- **Gradient Descent** to learn the model parameters
- **Mean Squared Error-based cost function** to measure model performance

The input and target variables are standardized before training.

## Model

The linear regression model is:

ŷ = wx + b

where:

- `w` is the weight
- `b` is the bias
- `x` is the input feature
- `ŷ` is the predicted value

The cost function used is:

J(w,b) = (1 / 2m) Σ (y⁽ⁱ⁾ - ŷ⁽ⁱ⁾)²

The gradients are calculated as:

∂J/∂w = (1/m) Σ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)x⁽ⁱ⁾

∂J/∂b = (1/m) Σ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)

The parameters are then updated using:

w := w - α(∂J/∂w)

b := b - α(∂J/∂b)

where `α` is the learning rate.

## Dataset

The project uses the **California Housing dataset** stored in:

`housing.csv`

The two variables used are:

- `median_income`
- `median_house_value`

Both variables are standardized before being used in gradient descent.

## Implementation

The implementation contains three main components.

### Cost Function

`compute_cost()` calculates the cost of the current model parameters.

### Gradient Calculation

`compute_gradient()` calculates the gradients of the cost function with respect to the weight and bias.

### Gradient Descent

`gradient_descent()` repeatedly updates the weight and bias using the calculated gradients.

The current experiment uses:

- Initial weight = `0`
- Initial bias = `0`
- Iterations = `2000`
- Learning rate = `0.05`

## Results

The project generates two main visualizations.

### Cost vs. Iteration

The cost decreases rapidly during the initial iterations and then becomes approximately constant, showing that gradient descent converges.

### Linear Regression

The second plot shows sampled actual observations together with the regression line obtained from gradient descent.

## Project Structure

```text
linear-regression-gradient-descent/
│
├── README.md
├── requirements.txt
├── GD_LR.py
├── housing.csv
└── Figure_1.png