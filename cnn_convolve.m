% Function: cnn_convolve()
% Date: 05/21/14
% ------------------------
% Computes all the convolved feature maps for a layer of a CNN, given the
% weights (convolution kernels) and biases learned by a patch-sized linear decoder.
%
% Convolving 100 images should take less than 3 minutes. Convolving 5000
% images should take about an hour.
%
% Inputs:
%
% patch_dim: dimension of the square patch over which the feature detector
%     (convolution kernel) was learned. 
%
% num_feats: number of convolved feature maps.
%
% images: [image dimension 1] x [image dimension 2] x [RGB channel] x [number
%     of images] matrix of images to extract convolved features from.
%
% W: [number of convolved features] x [number of RGB channels x patch_dim x
%     patch_dim] matrix of the weights i.e. convolution kernels learned by
%     a sparse autoencoder over patches of dimension patch_dim. 
%
%     i-th row gives the vectorized 2D kernels per channel of one feature,
%     concatenated as [i-th feature weights for R channel | i-th feature
%     weights for G channel | i-th feature weights for B channel], where
%     the vectorized feature weights for a channel can be reformatted as a
%     2D kernel by reshape(vectorized feature, patch_dim, patch_dim).
%
% b: num_feats bias terms learned by a sparse autoencoder.
%
% zca_white: ZCA whitening matrix used for preprocessing.
% 
% mean_patch: mean patch matrix used for preprocessing.
%
% Outputs:
%
% conv_feats: [number of features] x [number of images] x [dimension 1 of
%     feature map] x [dimension 2 of feature map] matrix of convolved
%     features

function conv_feats = cnn_convolve(patch_dim, num_feats, images, W, b, zca_white, mean_patch)

    num_im = size(images, 4);
    im_dim = size(images, 1);
    num_channels = size(images, 3);
    
    % Add preprocessing to the convolution kernels.
    conv_kernels = W * zca_white;
    b_preproc = b - conv_kernels * mean_patch;

    conv_feats = zeros(num_feats, num_im, im_dim - patch_dim + 1, im_dim - patch_dim + 1);
    for im_idx = 1 : num_im
        for feat_idx = 1 : num_feats
            
        % Convolve each channel with its convolution kernel for this
        % feature. The convolved feature of the image is the sum of the
        % convolved features of each channel.
        conv_im = zeros(im_dim - patch_dim + 1, im_dim - patch_dim + 1);
        for channel = 1 : num_channels

            % Extract the vectorized convolution kernel for this channel,
            % and reformat it as a 2D square. Flip since the mathematical
            % definition of convolution will flip back to original.
            feature_vec = conv_kernels(feat_idx, (channel - 1) * (patch_dim ^ 2) + 1 : ...
                channel * (patch_dim ^ 2));
            feature = reshape(feature_vec, patch_dim, patch_dim);
            feature = rot90(squeeze(feature),2);

            % Compute a valid convolution of the image channel with the kernel
            % and add to the convolved feature.
            im = squeeze(images(:, :, channel, im_idx));
            conv_im = conv_im + conv2(im, feature, 'valid');

        end
        
        % Add the bias for this feature and compute the nonlinearity for
        % the final convolved feature.
        conv_feats(feat_idx, im_idx, :, :) = sigmoid(conv_im + b_preproc(feat_idx));
        
      end
    end

end

% Function: sigmoid()
% Date: 05/22/14
% -------------------
% Computes the element-wise sigmoid function.

function sigm = sigmoid(x)

    sigm = 1 ./ (1 + exp(-x));

end
