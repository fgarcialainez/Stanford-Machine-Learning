function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Part 1 - Calculate Cost
predictions = X * theta;

sumItem = (predictions - y) .^ 2;
regItem = (sum(theta(2:size(theta)) .^ 2) .* lambda) ./ (2 * m);

J = (sum(sumItem) ./ (2 * m)) + regItem;


% Part 2 - Calculate Gradient
for j=1:size(grad)
	
	auxGrad = sum(X(:, j)' * (predictions - y)) ./ m;

	if(j > 1)	    
		regItem = (lambda * theta(j)) / m;
		auxGrad = auxGrad + regItem;
	end

	grad(j) = auxGrad;
end


% =========================================================================

grad = grad(:);

end
