function loss = compute_loss(y_pred, y_true)
m = size(y_true,1);
% Função de Perda adotada: Entropia Cruzada
loss = -sum(sum(y_true .* log(y_pred + 1e-8))) / m;
end
