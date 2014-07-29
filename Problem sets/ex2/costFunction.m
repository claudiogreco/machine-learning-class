function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% calculates the hypothesis withe the current theta
sig_predictions = sigmoid(X * theta);

% calculates the value of the cost function
J = (1/m) * sum(-y .* log(sig_predictions) - (1 - y) .* log(1 - sig_predictions));

for i = 1:size(theta, 1)
	% calculates the theta values for the gradient descent
	grad(i) = (1/m) * sum((sig_predictions - y) .* X(:, i));
endfor

% =============================================================

end
