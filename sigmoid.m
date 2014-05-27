% Function: sigmoid()
% Date: 05/26/14
% Author: Clara Fannjiang (clarafj@stanford.edu)
% ----------------------------------------------
% Implements the sigmoid function element-wise.

function sigm = sigmoid(x)

    sigm = 1 ./ (1 + exp(-x));
    
end