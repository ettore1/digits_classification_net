function visualize_mnist_samples(X, y, num_samples)
% VISUALIZE_MNIST_SAMPLES - Exibe amostras do dataset MNIST
%
% Uso:
%   visualize_mnist_samples(X, y, num_samples)
%
% Entradas:
%   X - matriz [n_samples x 784] (imagens achatadas e normalizadas)
%   y - rótulos one-hot [n_samples x 10] ou vetor [n_samples x 1]
%   num_samples - número de amostras a exibir (opcional, padrão = 25)
%
% Exemplo:
%   visualize_mnist_samples(X_train, y_train, 25)

    if nargin < 3
        num_samples = 25;
    end

    % Garante que não tentaremos exibir mais do que temos
    num_samples = min(num_samples, size(X,1));

    % Seleciona amostras aleatórias
    idx = randperm(size(X,1), num_samples);
    X_subset = X(idx, :);
    y_subset = y(idx, :);

    % Se os rótulos estiverem em one-hot, converte para inteiros
    if size(y_subset,2) > 1
        [~, y_subset] = max(y_subset, [], 2);
        y_subset = y_subset - 1; % classes de 0 a 9
    end

    % Determina grid de exibição
    grid_size = ceil(sqrt(num_samples));
    figure;
    colormap('gray');

    for i = 1:num_samples
        subplot(grid_size, grid_size, i);
        imshow(reshape(X_subset(i,:), [28, 28])', 'InitialMagnification', 'fit');
        title(sprintf('%d', y_subset(i)), 'FontSize', 10);
        axis off;
    end

    sgtitle(sprintf('Amostras MNIST (%d imagens normalizadas)', num_samples));
end
