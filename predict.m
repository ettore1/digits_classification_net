function y_pred_class = predict(X, net)
[y_pred, ~] = forward_pass(X, net);
[~, y_pred_class] = max(y_pred, [], 2);
end
