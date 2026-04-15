import numpy as np

class HandlerPerceptron:

    def __init__(self, num_features):
        # Inicialização dos parâmetros
        self.w = np.random.uniform(-1, 1, num_features)
        self.b = np.random.uniform(-1, 1)
        self.eta = np.random.uniform(0, 1)

    # ----------------------------------
    # Função de ativação (sinal)
    # ----------------------------------
    def funcao_sinal(self, u):
        return 1 if u >= 0 else -1

    # ----------------------------------
    # Cálculo da saída do perceptron
    # ----------------------------------
    def calcular_saida(self, x):
        u = np.dot(self.w, x) + self.b
        return self.funcao_sinal(u)

    # ----------------------------------
    # Atualização dos pesos
    # ----------------------------------
    def atualizar_pesos(self, x, d):
        print("\nAtualizando pesos...")

        self.w = self.w + self.eta * d * x
        self.b = self.b + self.eta * d

    # ----------------------------------
    # Debug (opcional, mas útil)
    # ----------------------------------
    def debug(self, x, y, d):
        print("\n--- DEBUG ---")
        print("Entrada:", x)
        print("Pesos:", self.w)
        print("Bias:", self.b)
        print("Taxa de aprendizado:", self.eta)
        print("Saída calculada (y):", y)
        print("Saída esperada (d):", d)
