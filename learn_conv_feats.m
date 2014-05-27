% Function: learn_conv_feats()
% Date: 05/26/14
% ---------------------------
% Learns patch feature detectors, i.e. convolution kernels for CNNs, from
% one-dimensional acoustic filter-bank features using a linear sparse
% autoencoder.

function opt_params = learn_conv_feats(train_patches, hidden_size, ...
    lambda, sparse_reg_param, sparse_target, max_lbfgs_iter, feat_file_name)

    patch_size = size(train_patches, 1);
    
    % patchDim = 8;
    % hidden_size  = 400;    % number of hidden units 
    % sparsityParam = 0.035; % desired average activation of the hidden units.
    % lambda = 3e-3;         % weight decay parameter       
    % sparse_reg_param = 5;  % weight of sparsity penalty term       
    % epsilon = 0.1;	     % epsilon for ZCA whitening
    
    % load stlSampledPatches.mat

    displayColorNetwork(train_patches(:, 1:100));

    % EXECUTE ZCA FILE AND SAVE PRE-PROCESSING MATRICES BEFORE THIS

    % Zero the mean of the training patches.
    % mean_patch = mean(train_patches, 2);  
    % train_patches = bsxfun(@minus, train_patches, mean_patch);

    % ZCA-whiten.
    % [u, s, ~] = svd(train_patches * train_patches' / numPatches);
    % zca_white = u * diag(1 ./ sqrt(diag(s) + zca_epsilon)) * u';
    % train_patches = zca_white * train_patches;
    
    % Initialize parameters randomly and run L-BFGS to train.
    params = init_params(hidden_size, patch_size);

    % Minimize the cost function using L-BFGS.
    addpath minFunc/
    options = struct;
    options.Method = 'lbfgs'; 
    options.maxIter = max_lbfgs_iter;
    options.display = 'on';

    opt_params = minFunc(@(p) sparse_ae_linear_cost(p, patch_size, hidden_size, ...
        lambda, sparse_reg_param, sparse_target, train_patches), params, options);

    % Save the learned convolution kernels.
    fprintf('Saving learned convolution kernels...\n');                          
    save(sprintf('%s.mat', feat_file_name), 'opt_params');
    fprintf('Saved.\n');
    
    % conv_feats = reshape(opt_params(1 : patch_size * hidden_size), hidden_size, patch_size);
    % displayColorNetwork((conv_feats * ZCAWhite)');

end
