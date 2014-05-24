% Function: softmax_cost()
% Date: 05/20/14
% ------------------------
% Computes the cost and gradient for softmax regression.
%
% num_class: number of classes.
% num_feats: number of features i.e. size of input.
% lambda: l2 regularization parameter.
% data: input features where each column gives one training instance.
% labels: ground truth labels.

function [cost, grad] = softmax_cost(theta, num_class, num_feats, lambda, data, labels)

    % Unroll the parameters from format needed for minFunc().
    theta = reshape(theta, num_class, num_feats);
    num_inst = size(data, 2);
    truth = full(sparse(labels, 1 : num_inst, 1));
    
    % Given the current parameters, compute the probability of each class
    % for each training instance.
    pre_hypoth = theta * data;
    unnorm_probs = exp(bsxfun(@minus, pre_hypoth, max(pre_hypoth)));
    probs = bsxfun(@rdivide, unnorm_probs, sum(unnorm_probs));
    log_probs = log(probs);
    
    % Compute the cost i.e. the loss plus l2 regularization.
    cost = (-1 / num_inst) * trace(log_probs' * truth) + ...
        (lambda / 2) * sum(sum(theta .^ 2));
    
    % Compute the gradient with respect to the parameters.
    theta_grad = (-1 / num_inst) * (truth - probs) * data' + lambda * theta;

    % Unroll the gradient matrices into a vector format for minFunc().
    grad = [theta_grad(:)];
    
end

