% Function: stacked_ae_predict()
% Date: 05/21/14
% ------------------------------
% Predicts labels given the parameters for DNN with stacked
% autoencoders into a softmax classifier, i.e. pred(i) = argmax_c P(y(c) | x(i)).
% Description of the "stack" structure is in stacked_ae_cost().
%
% Inputs:
%
% theta: trained autoencoder weights and softmax parameters of the DNN.
% most_hidden_size:  number of hidden units at the "most hidden"
%     autoencoder layer, whose outputs are features into the final softmax
%     classifier.
% num_classes: number of classes.
% data: one testing instance per column.

function pred = stackedAEPredict(theta, most_hidden_size, num_classes, ...
    net_config, data)
                                                                              
    % Extract (reformat) the softmax parameters and stacked autoencoder 
    % weights separately.
    softmax_theta = reshape(theta(1 : most_hidden_size * num_classes), ...
        num_classes, most_hidden_size);
    stack = params2stack(theta(most_hidden_size * num_classes + 1 : end), net_config);

    % Forward pass to compute the activations of each autoencoder layer.
    % (Consider the input data the activation of the first layer.)
    act_ae = [data cell(1, numel(stack))];
    for i = 1 : numel(stack)
        act_ae{i + 1} = sigmoid(stack{i}.w * act_ae{i} + repmat(stack{i}.b, 1, num_inst));
    end

    % Final (non-autoencoder) layer activations are the softmax classification 
    % output probabilities of each class. Don't need to exponentiate or
    % normalize since argmax is the same.
    [~, pred] = max(softmax_theta * act_ae{end});

end

% Function: sigmoid()
% Date: 05/21/14
% -------------------
% Element-wise sigmoid function.

function sigm = sigmoid(x)

    sigm = 1 ./ (1 + exp(-x));
    
end
