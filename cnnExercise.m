% File: cnn_exercise.m
% Date: 05/22/14
% ------------------
% CS294A/CS294W convolutional neural network exercise. Using the weights
% learned for a patch-sized sparse linear decoder, i.e. convolved feature
% kernels, implements convolved feature extraction and pooling, then uses
% the convolved features to train a softmax classifer.

%-----Initialize Parameters-----

im_dim = 64;
num_channels = 3;
patch_dim = 8;
num_patches = 50000;
pool_dim = 19; % Dimension of pooling square.

% Number of input units, outputs units, and hidden units (number of convolved
% features) in the sparse linear decoder.
input_size = patch_dim * patch_dim * num_channels;
output_size = input_size;
hidden_size = 400; 

% Epsilon for ZCA whitening.
epsilon = 0.1;

%-----Load and Display Weights-----

% Load the learned sparse linear decoder weights (and preprocessing
% matrices).
load STL10Features.mat optTheta ZCAWhite meanPatch
opt_theta = optTheta;
zca_white = zcaWhite;
mean_patch = meanPatch;

W = reshape(opt_theta(1 : input_size * hidden_size), hidden_size, input_size);
b = opt_theta(2 * hidden_size * input_size + 1 : 2 * hidden_size * input_size + hidden_size);
displayColorNetwork((W * zca_white)');

%%======================================================================
%% STEP 2: Implement and test convolution and pooling
%  In this step, you will implement convolution and pooling, and test them
%  on a small part of the data set to ensure that you have implemented
%  these two functions correctly. In the next step, you will actually
%  convolve and pool the features with the STL10 images.

%% STEP 2a: Implement convolution
%  Implement convolution in the function cnn_convolve in cnn_convolve.m

% Note that we have to preprocess the images in the exact same way 
% we preprocessed the patches before we can obtain the feature activations.

load stlTrainSubset.mat % loads numTrainImages, trainImages, trainLabels

%% Use only the first 8 images for testing
convImages = trainImages(:, :, :, 1:8); 
convolvedFeatures = cnn_convolve(patch_dim, hidden_size, convImages, W, b, zca_white, mean_patch);

%% STEP 2b: Checking your convolution
%  To ensure that you have convolved the features correctly, we have
%  provided some code to compare the results of your convolution with
%  activations from the sparse autoencoder

% For 1000 random points
for i = 1:1000    
    featureNum = randi([1, hidden_size]);
    imageNum = randi([1, 8]);
    imageRow = randi([1, im_dim - patch_dim + 1]);
    imageCol = randi([1, im_dim - patch_dim + 1]);    
   
    patch = convImages(imageRow:imageRow + patch_dim - 1, imageCol:imageCol + patch_dim - 1, :, imageNum);
    patch = patch(:);            
    patch = patch - mean_patch;
    patch = zca_white * patch;
    
    features = feedForwardAutoencoder(opt_theta, hidden_size, input_size, patch); 

    if abs(features(featureNum, 1) - convolvedFeatures(featureNum, imageNum, imageRow, imageCol)) > 1e-9
        fprintf('Convolved feature does not match activation from autoencoder\n');
        fprintf('Feature Number    : %d\n', featureNum);
        fprintf('Image Number      : %d\n', imageNum);
        fprintf('Image Row         : %d\n', imageRow);
        fprintf('Image Column      : %d\n', imageCol);
        fprintf('Convolved feature : %0.5f\n', convolvedFeatures(featureNum, imageNum, imageRow, imageCol));
        fprintf('Sparse AE feature : %0.5f\n', features(featureNum, 1));       
        error('Convolved feature does not match activation from autoencoder');
    end 
end

disp('Congratulations! Your convolution code passed the test.');

%% STEP 2c: Implement pooling
%  Implement pooling in the function cnn_pool in cnn_pool.m

% NOTE: Implement cnn_pool in cnn_pool.m first!
pooledFeatures = cnn_pool(pool_dim, convolvedFeatures);

%% STEP 2d: Checking your pooling
%  To ensure that you have implemented pooling, we will use your pooling
%  function to pool over a test matrix and check the results.

testMatrix = reshape(1:64, 8, 8);
expectedMatrix = [mean(mean(testMatrix(1:4, 1:4))) mean(mean(testMatrix(1:4, 5:8))); ...
                  mean(mean(testMatrix(5:8, 1:4))) mean(mean(testMatrix(5:8, 5:8))); ];
            
testMatrix = reshape(testMatrix, 1, 1, 8, 8);
        
pooledFeatures = squeeze(cnn_pool(4, testMatrix));

if ~isequal(pooledFeatures, expectedMatrix)
    disp('Pooling incorrect');
    disp('Expected');
    disp(expectedMatrix);
    disp('Got');
    disp(pooledFeatures);
else
    disp('Congratulations! Your pooling code passed the test.');
end

%%======================================================================
%% STEP 3: Convolve and pool with the dataset
%  In this step, you will convolve each of the features you learned with
%  the full large images to obtain the convolved features. You will then
%  pool the convolved features to obtain the pooled features for
%  classification.
%
%  Because the convolved features matrix is very large, we will do the
%  convolution and pooling 50 features at a time to avoid running out of
%  memory. Reduce this number if necessary

stepSize = 50;
assert(mod(hidden_size, stepSize) == 0, 'stepSize should divide hidden_size');

load stlTrainSubset.mat % loads numTrainImages, trainImages, trainLabels
load stlTestSubset.mat  % loads numTestImages,  testImages,  testLabels

pooledFeaturesTrain = zeros(hidden_size, numTrainImages, ...
    floor((im_dim - patch_dim + 1) / pool_dim), ...
    floor((im_dim - patch_dim + 1) / pool_dim) );
pooledFeaturesTest = zeros(hidden_size, numTestImages, ...
    floor((im_dim - patch_dim + 1) / pool_dim), ...
    floor((im_dim - patch_dim + 1) / pool_dim) );

tic();

for convPart = 1:(hidden_size / stepSize)
    
    featureStart = (convPart - 1) * stepSize + 1;
    featureEnd = convPart * stepSize;
    
    fprintf('Step %d: features %d to %d\n', convPart, featureStart, featureEnd);  
    Wt = W(featureStart:featureEnd, :);
    bt = b(featureStart:featureEnd);    
    
    fprintf('Convolving and pooling train images\n');
    convolvedFeaturesThis = cnn_convolve(patch_dim, stepSize, ...
        trainImages, Wt, bt, zca_white, mean_patch);
    pooledFeaturesThis = cnn_pool(pool_dim, convolvedFeaturesThis);
    pooledFeaturesTrain(featureStart:featureEnd, :, :, :) = pooledFeaturesThis;   
    toc();
    clear convolvedFeaturesThis pooledFeaturesThis;
    
    fprintf('Convolving and pooling test images\n');
    convolvedFeaturesThis = cnn_convolve(patch_dim, stepSize, ...
        testImages, Wt, bt, zca_white, mean_patch);
    pooledFeaturesThis = cnn_pool(pool_dim, convolvedFeaturesThis);
    pooledFeaturesTest(featureStart:featureEnd, :, :, :) = pooledFeaturesThis;   
    toc();

    clear convolvedFeaturesThis pooledFeaturesThis;

end


% You might want to save the pooled features since convolution and pooling takes a long time
save('cnn_pooledFeatures.mat', 'pooledFeaturesTrain', 'pooledFeaturesTest');
toc();

%%======================================================================
%% STEP 4: Use pooled features for classification
%  Now, you will use your pooled features to train a softmax classifier,
%  using softmaxTrain from the softmax exercise.
%  Training the softmax classifer for 1000 iterations should take less than
%  10 minutes.

% Add the path to your softmax solution, if necessary
% addpath /path/to/solution/

% Setup parameters for softmax
softmaxLambda = 1e-4;
numClasses = 4;
% Reshape the pooledFeatures to form an input vector for softmax
softmaxX = permute(pooledFeaturesTrain, [1 3 4 2]);
softmaxX = reshape(softmaxX, numel(pooledFeaturesTrain) / numTrainImages,...
    numTrainImages);
softmaxY = trainLabels;

options = struct;
options.maxIter = 200;
softmaxModel = softmaxTrain(numel(pooledFeaturesTrain) / numTrainImages,...
    numClasses, softmaxLambda, softmaxX, softmaxY, options);

%%======================================================================
%% STEP 5: Test classifer
%  Now you will test your trained classifer against the test images

softmaxX = permute(pooledFeaturesTest, [1 3 4 2]);
softmaxX = reshape(softmaxX, numel(pooledFeaturesTest) / numTestImages, numTestImages);
softmaxY = testLabels;

[pred] = softmaxPredict(softmaxModel, softmaxX);
acc = (pred(:) == softmaxY(:));
acc = sum(acc) / size(acc, 1);
fprintf('Accuracy: %2.3f%%\n', acc * 100);

% You should expect to get an accuracy of around 80% on the test images.
