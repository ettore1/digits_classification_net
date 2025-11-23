function [X_train, y_train, X_test, y_test] = load_mnist_subset(n_train, n_test)
    % LOAD_MNIST_SUBSET - Carrega subconjunto do MNIST (digit dataset)
    % e normaliza as imagens para [0,1].
    %
    % Saídas:
    %   X_train, X_test - imagens achatadas [n_samples x 784]
    %   y_train, y_test - rótulos one-hot [n_samples x 10]

    % Carrega os dados (funções internas do MATLAB)
    [XTrain, YTrain] = digitTrain4DArrayData;
    [XTest, YTest]   = digitTest4DArrayData;

    % Converter para [n_samples, 784]
    XTrain = reshape(XTrain, 28*28, [])';
    XTest  = reshape(XTest, 28*28, [])';

    % Normalizar para [0, 1]
    XTrain = double(XTrain);
    XTest  = double(XTest);

    % Converter rótulos em one-hot
    yTrain = onehotencode(YTrain, 2);
    yTest  = onehotencode(YTest, 2);

    % Selecionar subconjuntos menores (para rodar mais rápido)
    idx_train = randperm(size(XTrain,1), n_train);
    idx_test  = randperm(size(XTest,1), n_test);

    X_train = XTrain(idx_train, :);
    y_train = yTrain(idx_train, :);
    X_test  = XTest(idx_test, :);
    y_test  = yTest(idx_test, :);
end
