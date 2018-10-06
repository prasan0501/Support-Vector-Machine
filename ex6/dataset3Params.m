function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
Ctmp = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigmatmp = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

results = zeros(size(Ctmp,1)*size(sigmatmp,1),3);
row = 0;

for i=1:size(Ctmp,1)
  for j=1:size(sigmatmp,1) 
    row = row + 1;
    model= svmTrain(X, y, Ctmp(i,1), @(x1, x2) gaussianKernel(x1, x2, sigmatmp(j,1)));
    predictions = svmPredict(model, Xval);
    results(row,1) = Ctmp(i,1);
    results(row,2) = sigmatmp(j,1);
    results(row,3) = mean(double(predictions ~= yval));
  endfor
endfor

results = sortrows(results,3);

C = results(1,1);
sigma = results(1,2);

% =========================================================================

end
