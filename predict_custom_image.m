function predict_custom_image(image_path)
% PREDICT_CUSTOM_IMAGE - Carrega uma imagem (ex: número desenhado à mão)
% e usa o modelo MNIST salvo para predizer o dígito (0–9).

    % Verificar se o arquivo existe
    if ~isfile(image_path)
        error('Arquivo não encontrado: %s', image_path);
    end

    fprintf('🖼️  Carregando imagem: %s\n', image_path);

    % 1. Carregar o modelo
    if ~isfile('mnist_model.mat')
        error('Modelo não encontrado! Execute o treinamento primeiro (main.m).');
    end
    load('mnist_model.mat', 'net');

    % 2. Ler imagem e converter para grayscale
    img = imread(image_path);
    img = rgb2gray(img);
    if size(img,3) == 3
        img = rgb2gray(img);
    end

    % 3. Redimensionar para 28x28
    img = imresize(img, [28, 28]);

    % 4. Inverter cores se fundo for escuro (heurística simples)
    if mean(img(:)) < 128
        img = imcomplement(img);
    end

    % 5. Normalizar e achatar para vetor 1x784
    img_norm = double(img(:))' /255.0;

    % 6. Forward pass
    [y_pred, ~] = forward_pass(img_norm, net);

    % 7. Obter classe predita
    [~, pred_class] = max(y_pred, [], 2);
    predicted_digit = pred_class - 1; % ajustar para 0–9

    % 8. Exibir resultado
    figure;
    imshow(img, 'InitialMagnification', 'fit');
    title(sprintf('Dígito predito: %d', predicted_digit), 'FontSize', 14);
    fprintf('Predição: o modelo identificou o dígito como %d\n', predicted_digit);
end
