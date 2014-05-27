% Function: init_params()
% Date: 05/26/14
% ----------------------
% Initializes autoencoder weight and bias parameters randomly, based on the
% sizes of the visible and hidden layers.

function params = init_params(hidden_size, visible_size)

    % Use uniform random distribution on [-r, r].
    r  = sqrt(6) / sqrt(hidden_size+visible_size+1);
    W1 = rand(hidden_size, visible_size) * 2 * r - r;
    W2 = rand(visible_size, hidden_size) * 2 * r - r;
    b1 = zeros(hidden_size, 1);
    b2 = zeros(visible_size, 1);

    % Unrolls and concatenates all parameters into vector form, for use by
    % minFunc().
    params = [W1(:) ; W2(:) ; b1(:) ; b2(:)];

end

