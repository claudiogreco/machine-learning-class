function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

% retrieves the number of features
n = size(X_norm, 2);

% scroll among each feature
for i = 1:n

	% calculates the mean of the feature
	mu(1, i) = mean(X(:, i));

	% calculates the standard deviation of the feature
	sigma(1, i) = std(X(:, i));

% ends the for cycle
endfor

% scroll among each feature
for i = 1:n

	% subtracts the mean from every value of the feature
	X_norm(:, i) -= mu(i);

	% divides by the deviation every value of the feature
	X_norm(:, i) /= sigma(i);

% ends the for cycle
endfor

% ============================================================

end
