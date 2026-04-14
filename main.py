import numpy as np

# --------------------------------------------------
# Função de ativação (função sinal / degrau bipolar)
# Retorna:
#   1  se entrada >= 0
#  -1  se entrada < 0
# Melhoria: criar um handler de diferentes funções de ativação e passar a função escolhida para função de criação e teste do perceptron
# --------------------------------------------------
def funcao_sinal(valor_entrada):
    if valor_entrada >= 0:
        return 1
    else:
        return -1


# --------------------------------------------------
# Função que:
# 1. Lê os dados de entrada e saída
# 2. Inicializa pesos, bias e taxa de aprendizado
# 3. Calcula a saída do perceptron para UMA amostra
# 4. Compara com a saída esperada
#
# Retorna:
#   True  -> se classificou corretamente
#   False -> caso contrário
# --------------------------------------------------
def inicializar_e_testar_perceptron(caminho_arquivo_entrada,
                                    caminho_arquivo_saida,
                                    indice_do_perceptron):
    
    # ------------------------------
    # Leitura dos arquivos
    # ------------------------------
    matriz_entradas = np.loadtxt(
        caminho_arquivo_entrada,
        delimiter=",",
        encoding="utf-8-sig"
    )

    matriz_saidas = np.loadtxt(
        caminho_arquivo_saida,
        encoding="utf-8-sig"
    )


    # ------------------------------
    # Selecionar a coluna da saída
    # correspondente ao perceptron desejado
    #
    # Se o arquivo tiver várias colunas (uma por classe),
    # pegamos apenas uma delas
    # ------------------------------
    if matriz_saidas.ndim > 1:
        vetor_saidas = matriz_saidas[:, indice_do_perceptron]
    else:
        vetor_saidas = matriz_saidas

    # ------------------------------
    # Selecionar a primeira amostra
    # ------------------------------
    vetor_entrada = matriz_entradas[0]
    valor_saida_esperada = vetor_saidas[0]

    # ------------------------------
    # Descobrir automaticamente o número de entradas
    # (quantidade de características da letra)
    # ------------------------------
    quantidade_entradas = matriz_entradas.shape[1]

    # ------------------------------
    # Inicialização dos pesos
    # Valores aleatórios entre -1 e 1
    # ------------------------------
    vetor_pesos = np.random.uniform(-1, 1, quantidade_entradas)

    # ------------------------------
    # Inicialização do bias
    # ------------------------------
    bias = np.random.uniform(-1, 1)

    # ------------------------------
    # Inicialização da taxa de aprendizado
    # Valor entre (0, 1]
    # ------------------------------
    taxa_aprendizado = np.random.uniform(0, 1)

    # ------------------------------
    # Cálculo do potencial de ativação (u)
    # u = somatório(w * x) + bias
    # ------------------------------
    potencial_ativacao = np.dot(vetor_pesos, vetor_entrada) + bias

    # ------------------------------
    # Aplicação da função de ativação
    # ------------------------------
    valor_saida_calculada = funcao_sinal(potencial_ativacao)

    # ------------------------------
    # Exibição dos valores (debug)
    # ------------------------------
    print("\n--- DEBUG ---")
    print("Vetor de entrada:", vetor_entrada)
    print("Pesos iniciais:", vetor_pesos)
    print("Bias:", bias)
    print("Taxa de aprendizado:", taxa_aprendizado)
    print("Potencial de ativação (u):", potencial_ativacao)
    print("Saída calculada (y):", valor_saida_calculada)
    print("Saída esperada (d):", valor_saida_esperada)

    # ------------------------------
    # Comparação entre saída calculada e esperada
    # ------------------------------
    if valor_saida_calculada == valor_saida_esperada:
        return True
    else:
        return False


# --------------------------------------------------
# Função principal (main)
# --------------------------------------------------
def main():

    caminho_entrada = "/home/amandaventurin/Downloads/EP 1 IA/conjuntos de dados/caracteres-Fausett/caracteres-limpo.csv"
    caminho_saida = "/home/amandaventurin/Downloads/EP 1 IA/conjuntos de dados/caracteres-Fausett/saidasFausett.csv"

    # Índice do perceptron:
    # 0 -> primeira coluna (ex: letra 'a')
    # 1 -> segunda coluna (ex: 'b'), etc.
    indice_perceptron = 0

    resultado_classificacao = inicializar_e_testar_perceptron(
        caminho_entrada,
        caminho_saida,
        indice_perceptron
    )

    # ------------------------------
    # Resultado final
    # ------------------------------
    if resultado_classificacao:
        print("\nResultado: Classificação CORRETA")
    else:
        print("\nResultado: Classificação INCORRETA")


# --------------------------------------------------
# Execução do programa
# --------------------------------------------------
if __name__ == "__main__":
    main()
