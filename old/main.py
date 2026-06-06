from HandlerPerceptron import HandlerPerceptron
from LeituraArquivo import carregar_fausett

import numpy as np
import logging

from FuncoesAtivacao import FUNCOES_ATIVACAO

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

DEBUG = False
# sinal, sigmoid, tanh
nome_funcao = "sinal"
funcao_ativacao = FUNCOES_ATIVACAO[nome_funcao]


def treinar_perceptrons(X, Y, funcao_ativacao):

    MAX_EPOCAS = 1000

    lista_perceptrons = []

    
    # Percorre cada perceptron e inicializa peso, bias e taxa de aprendizado
    for indice_perceptron in range(Y.shape[1]):

        logger.info(
            f"Iniciando treinamento do perceptron {indice_perceptron}"
        )

        vetor_saidas = Y[:, indice_perceptron]


        perceptron = HandlerPerceptron(
            X.shape[1],
            funcao_ativacao
        )

        mudanca_neuronio = False
        epoca = 0

        
        # Epocas - continua até que o perceptron passe pelo loop interno sem mudar os pesos nenhuma vez
        while not mudanca_neuronio and epoca < MAX_EPOCAS:

            epoca += 1
            mudanca_neuronio = True

            logger.info(
                f"Perceptron {indice_perceptron} - Época {epoca}"
            )

            
            # Amostras - passa por cada amostra do conjunto tentando acertar o seu valor para aquela entrada
            for i in range(vetor_saidas.shape[0]):

                x = X[i]
                d = vetor_saidas[i]

                y = perceptron.calcular_saida(x)

                if DEBUG:
                    logger.info(
                        f"P={indice_perceptron} "
                        f"E={epoca} "
                        f"A={i} "
                        f"D={d} "
                        f"Y={y}"
                    )

                if y != d:

                    mudanca_neuronio = False

                    perceptron.atualizar_pesos(x, d)

                    if DEBUG:
                        logger.info(
                            f"Perceptron {indice_perceptron}: pesos atualizados"
                        )

        logger.info(
            f"Perceptron {indice_perceptron} treinado em {epoca} épocas"
        )

        lista_perceptrons.append(perceptron)

        if epoca == MAX_EPOCAS:

            logger.warning(
                f"Perceptron {indice_perceptron} atingiu "
                f"o limite de {MAX_EPOCAS} épocas."
            )

    return lista_perceptrons


def testar_perceptrons(lista_perceptrons, X_teste, Y_teste):

    logger.info("Iniciando fase de teste")

    total_amostras = X_teste.shape[0]
    total_acertos = 0

    # Amostras - passa por cada amostra do conjunto tentando acertar o seu valor para aquela entrada
    for indice_amostra in range(total_amostras):

        x = X_teste[indice_amostra]

        saidas_perceptrons = []

        # Percorre cada perceptron e calcula a saída
        for perceptron in lista_perceptrons:

            y = perceptron.calcular_saida(x)

            saidas_perceptrons.append(y)

        saidas_perceptrons = np.array(saidas_perceptrons)

        # classe prevista
        classe_predita = np.argmax(saidas_perceptrons)

        # classe correta
        classe_real = np.argmax(Y_teste[indice_amostra])

        if classe_predita == classe_real:
            total_acertos += 1

        if DEBUG:

            logger.info(
                f"Amostra={indice_amostra} "
                f"Classe Real={classe_real} "
                f"Classe Predita={classe_predita}"
            )

    acuracia = (
        total_acertos / total_amostras
    ) * 100

    logger.info(
        f"Acurácia final: {acuracia:.2f}%"
    )

    logger.info(
        f"Acertos: {total_acertos}/{total_amostras}"
    )

    return acuracia


def main():

    try:

        # TREINAMENTO
        caminho_treino = (
            "./conjuntos de dados/caracteres-Fausett/caracteres-limpo.csv"
        )

        X_treino, Y_treino = carregar_fausett(
            caminho_treino
        )

        logger.info(
            f"Dataset de treino carregado "
            f"({X_treino.shape[0]} amostras)"
        )

        lista_perceptrons = treinar_perceptrons(
            X_treino,
            Y_treino,
            funcao_ativacao
        )

        logger.info(
            f"{len(lista_perceptrons)} perceptrons treinados"
        )

        # TESTE

        caminho_teste = (
            "./conjuntos de dados/caracteres-Fausett/caracteres-ruido.csv"
        )

        X_teste, Y_teste = carregar_fausett(
            caminho_teste
        )

        logger.info(
            f"Dataset de teste carregado "
            f"({X_teste.shape[0]} amostras)"
        )

        testar_perceptrons(
            lista_perceptrons,
            X_teste,
            Y_teste
        )

        logger.info(
            "Execução finalizada com sucesso"
        )

    except Exception as erro:

        logger.exception(
            f"Erro durante execução: {erro}"
        )


if __name__ == "__main__":
    main()
