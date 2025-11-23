function predict_single_image(model_path)
    % PREDICT_SINGLE_IMAGE - Testa um modelo salvo do MNIST com uma imagem aleatória
    
    %% 1. Carregar modelo
    load(model_path, 'net');
    fprintf('Modelo carregado: %s\n', model_path);

    %% 2. Carregar dados de teste
    [~, ~, X_test, y_test] = load_mnist_subset(5000, 1000); % usa mesma função
    
    %% 3. Escolher imagem aleatória
    idx = randi(size(X_test, 1));
    x = X_test(idx, :);
    y_true_onehot = y_test(idx, :);
    [~, y_true] = max(y_true_onehot, [], 2);

    %% 4. Fazer predição
    [y_pred, ~] = forward_pass(x, net);
    [~, y_pred_class] = max(y_pred, [], 2);

    %% 5. Mostrar resultado
    fprintf('Índice: %d | Dígito verdadeiro: %d | Predito: %d\n', ...
        idx, y_true - 1, y_pred_class - 1);

    %% 6. Visualizar imagem
    img = reshape(x, [28, 28]);
    figure;
    imshow(img');
    title(sprintf('Verdadeiro: %d | Predito: %d', y_true - 1, y_pred_class - 1));
end
