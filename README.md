# digits_classification_net
Experimental Multi Layer Perceptron Neural Network to classify digits from MNIST Dataset

# Classificação de Dígitos Manuscritos com Rede Neural em MATLAB

Este projeto implementa **do zero**, em MATLAB, uma **rede neural multicamadas (MLP)** para classificação de dígitos escritos à mão utilizando o dataset **MNIST**.

Nenhuma toolbox de deep learning foi utilizada — todas as etapas foram programadas manualmente:
- inicialização dos pesos
- forward pass
- função sigmoide
- softmax
- função de perda cross-entropy
- backpropagation
- atualização dos parâmetros
- avaliação e visualização

---

## PRÉ-REQUISITOS
- MATLAB R2019+

*Funções internas*:
- digitTrain4DArrayData

- digitTest4DArrayData

obs: Essas funções já vêm em versões recentes do MATLAB. (https://www.mathworks.com/help/deeplearning/ug/data-sets-for-deep-learning.html)

---

## 📂 Estrutura do Projeto

📦 mnist-mlp-matlab
├── main.m # Script principal de treinamento da rede
├── load_mnist_subset.m # Carrega e pré-processa um subconjunto do MNIST
├── init_network.m # Inicializa os pesos e biases
├── forward_pass.m # Implementa a passagem direta
├── backpropagation.m # Cálculo dos gradientes via backprop
├── update_parameters.m # Atualiza os pesos com gradiente descendente
├── compute_loss.m # Calcula a cross-entropy
├── predict_custom_image.m # Predição para imagens desenhadas pelo usuário
├── predict_single_image.m # Predição de imagens do proprio repositório MNIST
├── visualize_predictions.m # Visualização de previsões individuais
├── mnist_model.mat # Modelo salvo após o treinamento
└── README.md # Este arquivo

---

## 🚀 Como Executar o Treinamento

Basta rodar o script principal:

```matlab
main
```

O script:

- Carrega 5000 imagens de treino e 1000 de teste do MNIST
- Treina a rede por N épocas
- Plota a curva de loss
- Exibe exemplos de predições
- Salva o modelo em mnist_model.mat

### 📊 Resultado do Treinamento

Após o treinamento, o script imprime:

- Acurácia no conjunto de teste (ex.: ~95%)
- Curva de perda (loss × épocas)
- Matriz de confusão (se habilitada)
- Visualização de algumas imagens corretamente e incorretamente classificadas

## 💾 Como Carregar o Modelo Salvo

A rede treinada é salva automaticamente como:

mnist_model.mat

Para carregá-lo em outro script:

```matlab
load mnist_model.mat
[y_pred, ~] = forward_pass(X, net);
```

## 🔍 Como Prever uma Imagem do Próprio MNIST

Caso queira visualizar uma previsão diretamente do dataset original, execute o arquivo indicando o caminho do modelo:

```matlab
predict_single_mnist_image('model_path')
```

O script apresenta o dígito e qual a probabilidade prevista.

## 🚀 Como Executar o Treinamento de Imagem Customizada

Você pode desenhar um dígito no Paint ou similar e salvar como .png, .jpg ou .bmp.

```matlab
predict_custom_image('meu_digito.png')
```

O script:
- Converte para grayscale
- Centraliza / redimensiona para 28×28
- Inverte se necessário (background branco)
- Normaliza para [0,1]
- Aplica o modelo carregado
- Mostra o dígito e a probabilidade prevista

- avaliação e visualização

---