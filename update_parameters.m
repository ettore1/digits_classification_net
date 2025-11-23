function net = update_parameters(net, grads, lr)
net.W1 = net.W1 - lr * grads.dW1;
net.b1 = net.b1 - lr * grads.db1;
net.W2 = net.W2 - lr * grads.dW2;
net.b2 = net.b2 - lr * grads.db2;
end
