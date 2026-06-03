import numpy as np

class Mlp:

    def __init__(self, n_inputs, n_hidden, n_outputs):
        # n_inputs: quantidade de entradas
        # n_hidden: quantidade de neurônios escondidos
        # n_outputs: quantidade de saídas
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        # Matriz V
        # Entrada e Camada
        # Dimensão: (n_hidden, n_inputs + 1)
        self.V = np.random.uniform(
            -0.5,
            0.5,
            (n_hidden, n_inputs + 1)
        )

        # Matriz W
        # Camada e Saida
        # Dimensão: (n_outputs, n_hidden + 1) +1 é o bias
        self.W = np.random.uniform(
            -0.5,
            0.5,
            (n_outputs, n_hidden + 1)
        )

    # Função de ativação
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivada_sigmoid(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    # até a camada escondida
    def forward_hidden(self, x):
        # adiciona bias na entrada
        x_bias = np.insert(x, 0, 1)

        # entrada --> camada
        z_in = np.dot(self.V, x_bias)

        # ativação
        z = self.sigmoid(z_in)

        return z

    # forward completo
    def forward(self, x):
        # camada escondida
        z = self.forward_hidden(x)

        # bias da camada escondida
        z_bias = np.insert(z, 0, 1)

        # camada de saída
        y_in = np.dot(self.W, z_bias)

        y = self.sigmoid(y_in)

        return y