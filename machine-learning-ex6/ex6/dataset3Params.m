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

%Train
#steps = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
C_vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

count = 0;
predictions = zeros(size(yval));
min_error = 1;

#results = zeros(length(C_list) * length(s_list), 3)

for temp_C = C_vals  
  for temp_sigma = sigma_vals    
    #model = svmTrain(X, y, temp_C, @(x1, x2) gaussianKernel(X(:,1), X(:,2), temp_sigma)); 
    #model = svmTrain(X, y, temp_C, @gaussianKernel(X(:,1), X(:,2), temp_sigma)); 
    model= svmTrain(X, y, temp_C, @(x1, x2) gaussianKernel(x1, x2, temp_sigma)); 
    %visualizeBoundary(X, y, model);
    predictions = svmPredict(model, Xval);
    pred_error = mean(double(predictions ~= yval));
    
    if pred_error < min_error
      min_error = pred_error;
      C = temp_C;
      sigma = temp_sigma;
    endif
    
    % save the results in the matrix
    #results(row,:) = [C_val sigma_val err_val]
    #row = row + 1;
    count += 1;
  endfor
endfor

#count




% =========================================================================

end
