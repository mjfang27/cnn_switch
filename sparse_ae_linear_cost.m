% Function: sparse_ae_linear_cost()
% Date: 05/21/14
% ----------------------------------
% Computes the objective and weight gradients for a linear decoder (sparse
% autoencoder whose output layer activation function is the identity,
% rather than the sigmoid function).

function [cost, grad] = sparse_ae_linear_cost(theta, visible_size, hidden_size, ...
    lambda, sparse_reg_param, sparse_target, data)

    % Convert from vector format (needed by minFunc()).
    W1 = reshape(theta(1 : hidden_size * visible_size), hidden_size, visible_size);
    W2 = reshape(theta(hidden_size * visible_size + 1 : 2 * hidden_size * visible_size), ...
        visible_size, hidden_size);
    b1 = theta(2 * hidden_size * visible_size + 1 : 2 * hidden_size * visible_size + hidden_size);
    b2 = theta(2 * hidden_size * visible_size + hidden_size + 1 : end);
    num_train = size(data, 2);

    % Forward pass to compute the activations of each unit, to be used in
    % both the objective and the gradient. 
    act_hidden = sigmoid(W1 * data + repmat(b1, 1, num_train));
    act_visible = W2 * act_hidden + repmat(b2, 1, num_train);
    
    % Compute sum of squared losses between visible activations (compressed images)
    % and data (original images).
    loss = (1 / (2 * num_train)) * (norm(act_visible - data, 'fro') ^ 2);

    % l2 regularization term on the weights.
    reg = (lambda / 2) * (norm(W1, 'fro') ^ 2 + norm(W2, 'fro') ^ 2);

    % Compute, for each hidden unit, the average activation over all the
    % training instances (rho-hat in the lecture notes).
    mean_act_hidden = mean(act_hidden, 2);

    % Sparsity-enforcing term, the KL divergence between Bernoulli random
    % variables with means of sparse_target and the observed average
    % activation of the hidden units.
    sparsity_enforce = sparse_target * (hidden_size * log(sparse_target) - sum(log(mean_act_hidden))) + ...
        (1 - sparse_target) * (hidden_size * log(1 - sparse_target) - sum(log(1 - mean_act_hidden)));

    cost = loss + reg + sparse_reg_param * sparsity_enforce;

    % Compute term in delta_hidden from sparsity enforcement term.
    sparsity_term = repmat((-sparse_target ./ mean_act_hidden) + ...
        (1 - sparse_target) ./ (1 - mean_act_hidden), 1, num_train);

    % Backward propagation. First compute the "error terms" describing how much each
    % unit contributes to the loss (delta in the lecture notes).
    delta_visible = act_visible - data;
    delta_hidden = (W2' * delta_visible + sparse_reg_param * sparsity_term) .* act_hidden .* (1 - act_hidden);

    % Compute the gradients of the loss term.
    W1_loss_grad = (delta_hidden * data') / num_train;
    W2_loss_grad = (delta_visible * act_hidden') / num_train;

    b1_grad = mean(delta_hidden, 2);
    b2_grad = mean(delta_visible, 2);

    W1_grad = W1_loss_grad + lambda * W1;
    W2_grad = W2_loss_grad + lambda * W2;

    % Convert to vector format for minFunc().
    grad = [W1_grad(:) ; W2_grad(:) ; b1_grad(:) ; b2_grad(:)];
    
end
