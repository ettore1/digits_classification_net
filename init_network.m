function net = init_network(n_input, n_hidden, n_output)
rng(0);

% Inicialização dos parâmetros inicial da rede com He (boa para ReLU)
net.W1 = randn(n_input, n_hidden) * sqrt(2 / n_input);
net.b1 = zeros(1, n_hidden);
net.W2 = randn(n_hidden, n_output) * sqrt(2 / n_hidden);
net.b2 = zeros(1, n_output);
end
