function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

	% predictions of hypothesis on all m examples
	hypothesis = X * theta;

	% errors: distance between predictions and examples
	errors = hypothesis - y;

	% retrieves the number of features
	n = size(X, 2);

	for i = 1:n

		% sets each theta value in the temporary vector
		theta_tmp(i, 1) = theta(i, 1) - alpha * (1/m) * sum(errors .* X(:, i));

	endfor

	% sets the final vector equal to the calculated vector
	theta = theta_tmp;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
