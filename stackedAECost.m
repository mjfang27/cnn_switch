% Function: stacked_ae_cost()
% Date: 05/20/14
% ---------------------------
% Computes the object and gradients of a two-layer stacked autoencoder.
% Used for fine-tuning DNN parameters after greedy single-layer training
% for initialization.
%
% The weights, biases, weight gradients, and bias gradients of autoencoder 
% layer l are stack{l}.w, stack{l}.b, stack_grad{l}.w and stack_grad{l}.b.
% The total number of autoencoder layers is numel(stack). The parameters
% and gradients for the final softmax classifer layer are stored
% separately in softmax_theta and softmax_theta_grad.
%
% Inputs:
%
% theta: greedy-trained weights for the autoencoder and softmax layers.
% most_hidden_size:  number of hidden units at the "most hidden"
%     autoencoder layer, right before the autoencoder layer.
% num_classes: number of classes in the softmax classifier.
% net_config: network configuration.
% lambda: l2 regularization parameter.
% data: one training instance per column.
% labels: ground truth class labels.

function [cost, grad] = stackedAECost(theta, most_hidden_size, num_classes, ...
    net_config, lambda, data, labels)

    num_inst = size(data, 2);
    truth = full(sparse(labels, 1 : num_inst, 1));

    % Extract the softmax classifier parameters and the stacked autoencoder
    % parameters separately.
    softmax_theta = reshape(theta(1 : most_hidden_size * num_classes), num_classes, most_hidden_size);
    stack = params2stack(theta(most_hidden_size * num_classes + 1 : end), net_config);
    num_ae_layers = numel(stack);
    
    % ACTIVATIONS
    
    % Forward pass to compute the activations of each autoencoder layer.
    % (Consider the input data the activation of the first layer.)
    act_ae = [data cell(1, num_ae_layers)];
    for i = 1 : num_ae_layers
        act_ae{i + 1} = sigmoid(stack{i}.w * act_ae{i} + repmat(stack{i}.b, 1, num_inst));
    end

    % Final (non-autoencoder) layer activations are the softmax classification 
    % output probabilities of each class.
    pre_hypoth = softmax_theta * act_ae{end};
    unnorm_probs = exp(bsxfun(@minus, pre_hypoth, max(pre_hypoth)));
    act_softmax = bsxfun(@rdivide, unnorm_probs, sum(unnorm_probs));
    
    % FINAL SOFTMAX CLASSIFIER COST AND GRADIENTS

    % Compute the whole DNN cost i.e. the log-likelihood plus l2 regularization.
    cost = (-1 / num_inst) * trace(log(act_softmax)' * truth) + ...
        (lambda / 2) * sum(sum(softmax_theta .^ 2));

    % Compute the gradient with respect to the softmax classifier parameters.
    softmax_theta_grad = (-1 / num_inst) * (truth - act_softmax) * act_ae{end}' + ...
        lambda * softmax_theta;
    
    % AUTOENCODER GRADIENTS
    
    % Compute the delta "error terms" of each autoencoder layer. For indexing
    % convenience, leave cell of first layer delta empty.
    delta = [cell(1, num_ae_layers) ...
        (-softmax_theta' * (truth - act_softmax)) .* act_ae{end} .* (1 - act_ae{end})];
    for i = fliplr(2 : num_ae_layers)
        delta{i} = (stack{i}.w' * delta{i + 1}) .* act_ae{i} .* (1 - act_ae{i});
    end

    % Compute the gradients with respect to the autoencoder weights and biases.
    stack_grad = cell(size(stack));
    for i = 1 : num_ae_layers
        stack_grad{i}.w = delta{i + 1} * act_ae{i}' / num_inst;
        stack_grad{i}.b = mean(delta{i + 1}, 2);
    end

    % Roll gradient vectors into format for minFunc().
    grad = [softmax_theta_grad(:); stack2params(stack_grad)];

end

% Function: sigmoid()
% Date: 05/20/14
% ------------------
% Element-wise sigmoid function.

function sigm = sigmoid(x)

    sigm = 1 ./ (1 + exp(-x));
    
end
