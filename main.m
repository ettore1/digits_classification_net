clear; clc; close all;
%% 1. Carregar subconjunto do MNIST
[X_train, y_train, X_test, y_test] = load_mnist_subset(5000, 1000);
fprintf('Tipo: %s | Classe: %s\n', mat2str(size(X_train)), class(X_train));
fprintf('Min: %.6f | Max: %.6f | mean: %.6f\n', min(X_train(:)), max(X_train(:)), mean(X_train(:)));

%% 2. Arquitetura
n_input = size(X_train, 2);
n_hidden = 64;
n_output = 10;
lr = 0.05;
epochs = 2000;

%% 3. Inicializar rede
net = init_network(n_input, n_hidden, n_output);

%% 4. Treinamento
losses = zeros(epochs,1);
train_acc = zeros(epochs,1);   % << novo

for epoch = 1:epochs
    [y_pred, cache] = forward_pass(X_train, net);

    if epoch == 1
        fprintf('\n--- Diagnóstico inicial ---\n');
        fprintf('Mean(Z1): %.4f | Std(Z1): %.4f\n', mean(cache.Z1(:)), std(cache.Z1(:)));
        fprintf('Mean(A1): %.4f | Std(A1): %.4f\n', mean(cache.A1(:)), std(cache.A1(:)));
        fprintf('Mean(y_pred): %.4f | Std(y_pred): %.4f\n', mean(y_pred(:)), std(y_pred(:)));
        fprintf('Sum(y_pred row 1): %.4f\n', sum(y_pred(1,:)));
    end

    loss = compute_loss(y_pred, y_train);
    grads = backpropagation(net, cache, y_train);
    net = update_parameters(net, grads, lr);
    losses(epoch) = loss;

    % --- Cálculo da acurácia de treinamento ---
    [~, pred_train_class] = max(y_pred, [], 2);
    [~, true_train_class] = max(y_train, [], 2);
    acc_train = mean(pred_train_class == true_train_class);
    train_acc(epoch) = acc_train;

    if mod(epoch,5)==0
        fprintf('Época %d/%d - Loss: %.4f - Acurácia treino: %.2f%%\n', ...
            epoch, epochs, loss, 100*acc_train);
    end
end

%% 5. Avaliação
[y_pred_test, ~] = forward_pass(X_test, net);
[~, y_pred_class] = max(y_pred_test, [], 2);
[~, y_true_class] = max(y_test, [], 2);

acc = mean(y_pred_class == y_true_class);
fprintf('\nAcurácia no conjunto de teste: %.2f%%\n', 100*acc);

%% 6. Visualizações
figure;
plot(losses, 'LineWidth', 1.5);
xlabel('Época'); ylabel('Loss'); title('Curva de Treinamento');

figure;
plot(train_acc, 'LineWidth', 1.5);
xlabel('Época'); ylabel('Acurácia');
title('Acurácia no Treinamento ao longo das épocas');
ylim([0 1]);

figure;
confusionchart(y_true_class, y_pred_class);
title('Matriz de Confusão - Teste');


save('mnist_model.mat', 'net');
