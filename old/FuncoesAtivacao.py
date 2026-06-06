import numpy as np


def funcao_sinal(x):

    if x >= 0:
        return 1

    return -1


def sigmoid(x):

    return 1 / (1 + np.exp(-x))


def tanh(x):

    return np.tanh(x)

FUNCOES_ATIVACAO = {
    "sinal": funcao_sinal,
    "sigmoid": sigmoid,
    "tanh": tanh
}
