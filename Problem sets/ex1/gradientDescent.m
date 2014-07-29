function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
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
    %       of the cost function (computeCost) and gradient here.
    %

	% predictions of hypothesis on all m examples
	hypothesis = X * theta;

	% errors: distance between predictions and examples
	errors = hypothesis - y;

	% calculates the current theta 0
	tempTheta0 = theta(1) - alpha * (1/m) * sum(errors);

	% calculates the current theta 1
	tempTheta1 = theta(2) - alpha * (1/m) * sum(errors .* X(:, 2));

	% sets the theta 0 value
	theta(1) = tempTheta0;

	% sets the theta 1 value
	theta(2) = tempTheta1;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
