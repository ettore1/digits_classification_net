function [y_pred, cache] = forward_pass(X, net)
Z1 = X * net.W1 + net.b1;
A1 = max(0, Z1); % ReLU
Z2 = A1 * net.W2 + net.b2;

% Softmax aplicada na saída da rede
exp_scores = exp(Z2 - max(Z2, [], 2));
y_pred = exp_scores ./ sum(exp_scores, 2);

cache.X = X;
cache.Z1 = Z1;
cache.A1 = A1;
cache.Z2 = Z2;
cache.y_pred = y_pred;
end
