import numpy as np

class HandlerPerceptron:

    def __init__(
        self,
        quantidade_entradas,
        funcao_ativacao
    ):

        self.w = np.random.uniform(
            -1,
            1,
            quantidade_entradas
        )

        self.b = np.random.uniform(-1, 1)

        self.eta = np.random.uniform(0, 1)

        self.funcao_ativacao = funcao_ativacao


    # ----------------------------------
    # Função de ativação (sinal)
    # ----------------------------------
    def calcular_saida(self, x):

        u = np.dot(self.w, x) + self.b

        return self.funcao_ativacao(u)


    # ----------------------------------
    # Atualização dos pesos
    # ----------------------------------
    def atualizar_pesos(self, x, d):
        print("\nAtualizando pesos...")

        self.w = self.w + self.eta * d * x
        self.b = self.b + self.eta * d

    # ----------------------------------
    # Debug
    # ----------------------------------
    def debug(self, x, y, d):
        print("\n--- DEBUG ---")
        print("Entrada:", x)
        print("Pesos:", self.w)
        print("Bias:", self.b)
        print("Taxa de aprendizado:", self.eta)
        print("Saída calculada (y):", y)
        print("Saída esperada (d):", d)
