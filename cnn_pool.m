% Function: cnn_pool()
% Date: 05/22/14
% --------------------
% Compute the matrix of mean-pooled convolved features for each convolved
% feature, image in the set, and image patch.
%
% Inputs:
%
% pool_dim: dimension of the square image patch to pool.
% conv_feats: [number of features] x [number of images] x [dimension 1 of
%     feature map] x [dimension 2 of feature map] matrix of convolved
%     features of the image set output by cnn_convolve().
%
% Outputs:
%
% pooled_feats: [number of features] x [number of images] x [dimension 1 of
%     pooled features, i.e. number of square pools along dimension 1 of
%     feature map] x [dimension 2 of pooled features] matrix of pooled
%     features.

function pooled_feats = cnn_pool(pool_dim, conv_feats)  

    pooled_feat_dim = floor(size(conv_feats, 3) / pool_dim);
    pooled_feats = zeros(size(conv_feats, 1), size(conv_feats, 2), ...
        pooled_feat_dim, pooled_feat_dim);
    
    % Summation pool.
    for i = 1 : pooled_feat_dim
        for j = 1 : pooled_feat_dim
            pooled_feats(:, :, i, j) = ...
                sum(sum(conv_feats(:, :, (i - 1) * pool_dim + 1 : i * pool_dim, ...
                (j - 1) * pool_dim + 1 : j * pool_dim), 4), 3);
        end
    end
    
    % Take mean.
    pooled_feats = pooled_feats / (pool_dim ^ 2);

end

