function visualize_predictions(X_test, y_true, y_pred)
idx = randperm(size(X_test,1), 9);
figure;
for i = 1:9
    subplot(3,3,i);
    imshow(reshape(X_test(idx(i),:), [28,28])', []);
    title(sprintf('Verdadeiro: %d | Previsto: %d', y_true(idx(i)), y_pred(idx(i))));
end
sgtitle('Predições da Rede Neural');
end
