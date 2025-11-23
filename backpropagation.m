function grads = backpropagation(net, cache, y_true)
m = size(y_true,1);

dZ2 = cache.y_pred - y_true;
dW2 = (cache.A1' * dZ2) / m;
db2 = sum(dZ2, 1) / m;

dA1 = dZ2 * net.W2';
dZ1 = dA1 .* (cache.Z1 > 0); % derivada da ReLU
dW1 = (cache.X' * dZ1) / m;

% calculo do gradiente descendente estocástico (SGD)
db1 = sum(dZ1, 1) / m;

grads.dW1 = dW1;
grads.db1 = db1;
grads.dW2 = dW2;
grads.db2 = db2;
end
