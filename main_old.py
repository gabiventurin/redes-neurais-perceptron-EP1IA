from HandlerPerceptron import HandlerPerceptron
import numpy as np


def main():

    caminho_entrada = "./conjuntos de dados/caracteres-Fausett/caracteres-limpo.csv"
    caminho_saida = "./conjuntos de dados/caracteres-Fausett/saidasFausett.csv"

    # ------------------------------
    # Leitura dos dados
    # ------------------------------
    matriz_entradas = np.loadtxt(
        caminho_entrada,
        delimiter=",",
        encoding="utf-8-sig"
    )

    matriz_saidas = np.loadtxt(
        caminho_saida,
        encoding="utf-8-sig"
    )

    # ---------------------------------------------------
    # Lista para armazenar os perceptrons treinados
    # ---------------------------------------------------
    lista_perceptrons = []

    # ===================================================
    # PRIMEIRO LAÇO
    # Itera sobre os perceptrons/classes
    # ===================================================
    for indice_perceptron in range(matriz_saidas.shape[1]):

        print("\n==================================================")
        print(f"INICIANDO TREINAMENTO DO PERCEPTRON {indice_perceptron}")
        print("==================================================")

        # ---------------------------------------------------
        # Seleciona coluna correta da saída
        # Cada perceptron aprende apenas sua classe
        # ---------------------------------------------------
        if matriz_saidas.ndim > 1:
            vetor_saidas = matriz_saidas[:, indice_perceptron]
        else:
            vetor_saidas = matriz_saidas

        # ---------------------------------------------------
        # Inicialização do perceptron
        # ---------------------------------------------------
        num_features = matriz_entradas.shape[1]

        perceptron = HandlerPerceptron(num_features)

        # ---------------------------------------------------
        # Controle das épocas
        # ---------------------------------------------------
        mudanca_neuronio = False

        epoca = 0

        # ===================================================
        # SEGUNDO LAÇO
        # Itera sobre as épocas
        # ===================================================
        while not mudanca_neuronio:

            epoca += 1

            print("\n--------------------------------------------------")
            print(f"ÉPOCA {epoca}")
            print("--------------------------------------------------")

            mudanca_neuronio = True

            # ===================================================
            # TERCEIRO LAÇO
            # Itera sobre as amostras
            # ===================================================
            for i in range(vetor_saidas.shape[0]):

                x = matriz_entradas[i]
                d = vetor_saidas[i]

                # ---------------------------------------------------
                # Forward
                # ---------------------------------------------------
                y = perceptron.calcular_saida(x)

                # ---------------------------------------------------
                # DEBUG COMPLETO
                # ---------------------------------------------------
                print("\n================ DEBUG ================")

                print(f"Perceptron atual: {indice_perceptron}")
                print(f"Época atual: {epoca}")
                print(f"Amostra atual: {i}")

                print("\nEntrada (x):")
                print(x)

                print("\nSaída esperada (d):")
                print(d)

                print("\nSaída calculada (y):")
                print(y)

                print("\nPesos atuais:")
                print(perceptron.w)

                print("\nBias atual:")
                print(perceptron.b)

                print("\nTaxa de aprendizado:")
                print(perceptron.eta)

                # ---------------------------------------------------
                # Atualização dos pesos
                # ---------------------------------------------------
                if y != d:

                    print("\n>>> ERRO ENCONTRADO")
                    print(">>> Atualizando pesos...")

                    mudanca_neuronio = False

                    perceptron.atualizar_pesos(x, d)

                    print("\nNovos pesos:")
                    print(perceptron.w)

                    print("\nNovo bias:")
                    print(perceptron.b)

                    print("\nResultado: Classificação INCORRETA")

                else:

                    print("\nResultado: Classificação CORRETA")

        # ---------------------------------------------------
        # Final do treinamento do perceptron
        # ---------------------------------------------------
        print("\n==================================================")
        print(f"PERCEPTRON {indice_perceptron} TREINADO")
        print("==================================================")

        print("\nPesos finais:")
        print(perceptron.w)

        print("\nBias final:")
        print(perceptron.b)

        print(f"\nTotal de épocas utilizadas: {epoca}")

        # ---------------------------------------------------
        # Salva o perceptron treinado
        # ---------------------------------------------------
        lista_perceptrons.append(perceptron)

    # ===================================================
    # RESULTADO FINAL GERAL
    # ===================================================
    print("\n\n##################################################")
    print("RESULTADO FINAL DE TODOS OS PERCEPTRONS")
    print("##################################################")

    for i, perceptron in enumerate(lista_perceptrons):

        print("\n==================================================")
        print(f"PERCEPTRON {i}")
        print("==================================================")

        print("\nPesos finais:")
        print(perceptron.w)

        print("\nBias final:")
        print(perceptron.b)

        print("\nTaxa de aprendizado:")
        print(perceptron.eta)


if __name__ == "__main__":
    main()
