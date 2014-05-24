% Function: compute_numerical_grad(J, theta)
% Date: 05/15/14
% ------------------------------------------
% Computes a fast approximation of the gradient of the function J
% at theta, by perturbing theta along each axes and using the limit
% definition of the gradient.

function num_grad = compute_numerical_grad(J, theta)
  
    EPSILON = 1e-4;
    
    num_grad = zeros(size(theta));
    for i = 1 : numel(theta)
        incr_by_eps = theta;
        incr_by_eps(i) = incr_by_eps(i) + EPSILON;
        decr_by_eps = theta;
        decr_by_eps(i) = decr_by_eps(i) - EPSILON;
        num_grad(i) = (J(incr_by_eps) - J(decr_by_eps)) / (2 * EPSILON);
    end

end
