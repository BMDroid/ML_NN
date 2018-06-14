function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1)); % 25 * 401

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); % 10 * 26

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m, 1) X];
a1 = X; % 5000 * 401
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2]; % 5000 * 26
z3 = a2 * Theta2';
a3 = sigmoid(z3); % 5000 * 10
h_theta = a3;

y_i = zeros(num_labels, 1); % the y(i) in column vector
y_binary  = zeros(m, num_labels); % 5000 * 10
cost = 0;

for i = 1:m
  y_i(y(i)) = 1;
  cost += log(h_theta(i,:)) * y_i + log(1 .- h_theta(i,:)) * (1 .- y_i);
  y_binary(i,:) = y_i';
  y_i = zeros(num_labels, 1);
endfor

Theta1_temp = Theta1(:, 2:end);
Theta2_temp = Theta2(:, 2:end);

J = -1 / m * cost + lambda / (2 * m) * (sum(Theta1_temp(:) .* Theta1_temp(:)) + sum(Theta2_temp(:) .* Theta2_temp(:)));

delta_L = a3 - y_binary; % 5000 * 10
Delta_2 = zeros(num_labels, hidden_layer_size + 1); % 10 * 26, accumulator for Theta2_grad
Delta_1 = zeros(hidden_layer_size, input_layer_size + 1); % 25 * 401, accumulator for Theta1_grad

for i = 1:m
  Delta_2 += delta_L(i, :)' * a2(i, :);
  delta_2 = delta_L(i, :) * Theta2 .* a2 .* (1 - a2);
  delta_2 = delta_2(:,2:end); % remove the term associated with the bias
  Delta_1 += delta_2(i, :)' * a1(i, :);
endfor

Theta1_temp = Theta1;
Theta1_temp(:, 1) = 0;

Theta2_temp = Theta2;
Theta2_temp(:, 1) = 0; 

Theta2_grad = 1 / m * Delta_2 + lambda / m * Theta2_temp;
Theta1_grad = 1 / m * Delta_1 + lambda / m * Theta1_temp;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end