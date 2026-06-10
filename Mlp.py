import numpy as np
import time

class Mlp:

    def __init__(self, n_inputs, n_hidden, n_outputs, alpha):
        self.n_inputs = n_inputs # quantidade de entradas
        self.n_hidden = n_hidden # quantidade de neurônios escondidos
        self.n_outputs = n_outputs # quantidade de saídas
        self.alpha = alpha

        # Matriz V - pesos e bias da Entrada e Camada Oculta
        # Dimensão: (n_hidden, n_inputs + 1)
        self.V = np.random.uniform(
            -1,
            1,
            (n_hidden, n_inputs + 1)
        )

        # Matriz W - pesos e bias da Camada e Saida
        # Dimensão: (n_outputs, n_hidden + 1) +1 é o bias
        self.W = np.random.uniform(
            -1,
            1,
            (n_outputs, n_hidden + 1)
        )

    # Função de ativação - ANTIGA
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivada_sigmoid(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    # Função de ativação 
    # tanh retorna valores em (-1, 1), compatível com saídas bipolares
    def tanh(self, x):
        return np.tanh(x)

    # derivada da tanh: 1 - tanh(x)²
    def derivada_tanh(self, x):
        return 1 - np.tanh(x) ** 2

    # até a camada escondida
    def forward_hidden(self, x):
        # adiciona bias na entrada
        x_bias = np.insert(x, 0, 1)

        # entrada --> camada escondida
        z_in = np.dot(self.V, x_bias)

        # ativação
        z = self.tanh(z_in)

        return x_bias, z_in, z


    # forward completo
    def forward(self, x):
        x_bias, z_in, z = self.forward_hidden(x)

        # camada escondida com bias
        z_bias = np.insert(z, 0, 1)

        # camada de saída
        y_in = np.dot(self.W, z_bias)
        y = self.tanh(y_in)

        return x_bias, z_in, z_bias, y_in, y


    def backpropagation(self, x_bias, z_in, z_bias, y_in, y, saidas_esperadas):
        # matrizes de correção
        WCorr = np.zeros_like(self.W)
        VCorr = np.zeros_like(self.V)

        
        ## BP PARTE 1: calcula os deltas (correções ou termo de informação de erro) para o vetor W 
        delta_k = np.zeros(self.n_outputs)
        for k in range(self.n_outputs):

            delta_k[k] = (
                (saidas_esperadas[k] - y[k])
                * self.derivada_tanh(y_in[k])
            )

            # bias (j = 0)
            WCorr[k, 0] = self.alpha * delta_k[k]

            # pesos ligados aos neurônios escondidos
            for j in range(self.n_hidden):
                WCorr[k, j + 1] = (
                    self.alpha
                    * delta_k[k]
                    * z_bias[j+1]
                )


        ## BP PARTE 2: calcula os deltas (correções) para o vetor V ##
        for j in range(self.n_hidden):

            delta_in_j = 0

            #calculando o 'termo de informação' de cada neuronio escondido, ponderado pelos pesos referentes a eles
            for k in range(self.n_outputs):

                delta_in_j += (
                    delta_k[k]
                    * self.W[k, j + 1] # j+1 porque coluna 0 é bias
                )

            #calcula o delta do neurônio escondido j, multiplicando o termo de informação pelo valor da derivada da função de ativação naquele ponto
            delta_j = (
                delta_in_j
                * self.derivada_tanh(z_in[j])
            )

            # correção do bias
            VCorr[j, 0] = self.alpha * delta_j

            # correção dos demais pesos
            for i in range(self.n_inputs):

                VCorr[j, i + 1] = (
                    self.alpha
                    * delta_j
                    * x_bias[i+1]
                )


        ## BP PARTE 3: atualiza os pesos ##
        self.W += WCorr
        self.V += VCorr
    
    def train(self, X, T, epocas=1000, erro_minimo=1e-4, X_val=None, T_val=None, patience=10): 
        """
        X, T            : dados de treino
        X_val, T_val    : dados de validação (optional)
                          Se fornecidos, ativa early stopping.
        patience        : épocas sem melhora no val antes de parar
        """
        historico_erro = []

        usar_val = X_val is not None and T_val is not None
        melhor_erro_val = np.inf
        epocas_sem_melhora = 0
        melhores_pesos = None

        start_time = time.time()

        for epoca in range(epocas):
            erro_total = 0.0

            for x, t in zip(X, T):
                x_bias, z_in, z_bias, y_in, y = self.forward(x)
                self.backpropagation(x_bias, z_in, z_bias, y_in, y, t)

                # Erro quadrático da amostra: 0.5 * sum((t - y)²)
                erro_total += 0.5 * np.sum((t - y) ** 2)

            erro_medio = erro_total / len(X)
            historico_erro.append(erro_medio)

            # avaliação no conjunto de validação
            if usar_val:
                erro_val = np.mean([
                    0.5 * np.sum((t - self.predict(x)) ** 2)
                    for x, t in zip(X_val, T_val)
                ])

                if erro_val < melhor_erro_val:
                    melhor_erro_val = erro_val
                    epocas_sem_melhora = 0
                    melhores_pesos = (self.W.copy(),
                                      self.V.copy())
                else:
                    epocas_sem_melhora += 1


            if (epoca + 1) % 100 == 0:
                end_time = time.time()
                msg = f"Época {epoca + 1:5d} | Erro médio: {erro_medio:.6f}"
                if usar_val:
                    msg += f" | Erro val: {erro_val:.6f}" 
                print(msg + f" | Tempo: {end_time - start_time:.2f}s")

            if erro_medio <= erro_minimo:
                end_time = time.time()
                print(f"Convergiu na época {epoca + 1} com erro {erro_medio:.6f}"
                      f" | Tempo total: {end_time - start_time:.2f}s")
                break

            # critério de parada antecipada
            if usar_val and epocas_sem_melhora >= patience:
                end_time = time.time()
                print(f"Early stopping na época {epoca + 1} "
                      f"(sem melhora por {patience} épocas) "
                      f"| Melhor erro val: {melhor_erro_val:.6f}"
                      f" | Tempo total: {end_time - start_time:.2f}s")
                self.W, self.V = melhores_pesos
                break

        # restaura melhores pesos caso tenha encerrado pelo erro_minimo ou epocas
        if usar_val and melhores_pesos is not None:
            self.W, self.V = melhores_pesos

        return historico_erro, melhor_erro_val
    

    def predict(self, x):

        _, _, _, _, y = self.forward(x)

        return y
