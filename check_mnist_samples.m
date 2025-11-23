clear; clc; close all;
[X_train, y_train, X_test, y_test] = load_mnist_subset(5000, 1000);
visualize_mnist_samples(X_train, y_train, 25);