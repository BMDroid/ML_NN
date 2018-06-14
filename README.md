# ML_NN
Neural Networks and Backpropagation Algorithm in Matlab.

## Neural Networks

From the previous project we could see that the linear regression and the logistic regression has pretty well results, but they cannot deal with the more complex hypotheses. However, the neural networks will be able to  represent **complex models** that form **non-linear** hypotheses. 

Here, I implemented the neural network to recognize the handwritten digits from 0 to 9. 

### 1. Data Visualization

<p align="center">  <img src="https://github.com/BMDroid/ML_NN/blob/master/figs/data.png" width="50%">
</p>  

### 2. Model representation

The structure of the NN in the project contains 3 layers:

- One input layer
- One hidden layer
- One output layer

And the initialized the Theta1 and Theta2 which fit the model are stored in the ex3weights.mat.

<p align="center">  <img src="https://github.com/BMDroid/ML_NN/blob/master/figs/model.png" width="50%">
</p>  

### 3. Feedforward  Propagation and Prediction

In the predict.m, we use the following codes to return the neural network’s predictions.

```matlab
X = [ones(m, 1) X]; % the features of the data
z2 = X * Theta1'; % first layer
a2 = sigmoid(z2); % the output of the input layer
a2 = [ones(m, 1) a2] % add ther bias
z3 = a2 * Theta2';
a3 = sigmoid(z3); % the final output should contains m rows and K columns
[M, I] = max(a3, [], 2); % pick the maximum elements in each row and show its indice
p = I; & P is the prediction of the digit
```

### 4. Results

The accuracy of the NN for handwritten digits is about 97.5%.

---

### 5. The Feedforward and cost function

The **cost function** of the neural networks is:

<p align="center">  <img src="https://github.com/BMDroid/ML_NN/blob/master/figs/cost_func.png" width="45%">
</p>  

And the **regularized** cost function is the following:

<p align="center">  <img src="https://github.com/BMDroid/ML_NN/blob/master/figs/cost_func_r.png" width="45%">
</p>  

```matlab
Theta1_temp = Theta1(:, 2:end);
Theta2_temp = Theta2(:, 2:end);

J = -1 / m * cost + lambda / (2 * m) * (sum(Theta1_temp(:) .* Theta1_temp(:)) + sum(Theta2_temp(:) .* Theta2_temp(:)));
```

### 6. Backpropagation

In the first of this project, we have the trained Theta1 and Theta2 for our hypotheses. Now we need find a way to compute them.

We need to use the **Backpropagation** Algorithm to compute the gradient for the NN cost function.

<p align="center">  <img src="https://github.com/BMDroid/ML_NN/blob/master/figs/backprop.png" width="50%">
</p>  

- For backpropgation, we need random initialize the weights.

  ```matlab
  epsilon_init = 0.12;
  W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
  ```

  

- And we use the **fmincg** to compute the gradient descent

### 7. Visualize the hidden layer

<p align="center">  <img src="https://github.com/BMDroid/ML_NN/blob/master/figs/hidden.png" width="50%">
</p>  

Training Set Accuracy: 96.200000