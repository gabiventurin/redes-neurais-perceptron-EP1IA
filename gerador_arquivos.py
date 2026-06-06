import os
import numpy as np

def salvar_hiperparametros(pasta_saida, n_inputs, n_hidden, n_outputs, alpha, epocas, erro_min, patience, shape_v, shape_w):
    """Gera o arquivo 1: Hiperparâmetros da arquitetura"""
    caminho_hiper = os.path.join(pasta_saida, "hiperparametros.txt")
    with open(caminho_hiper, "w", encoding="utf-8") as f:
        f.write("===== HIPERPARÂMETROS DA REDE NEURAL MLP =====\n\n")
        f.write("[Arquitetura]\n")
        f.write(f"  n_inputs  = {n_inputs}\n")
        f.write(f"  n_hidden  = {n_hidden}\n")
        f.write(f"  n_outputs = {n_outputs}\n\n")
        f.write("[Treinamento]\n")
        f.write(f"  alpha (taxa de aprendizado) = {alpha}\n")
        f.write(f"  epocas (máximo)             = {epocas}\n")
        f.write(f"  erro_minimo                 = {erro_min}\n")
        f.write(f"  patience (early stopping)   = {patience}\n\n")
        f.write("[Função de ativação]\n")
        f.write("  Camada escondida : tanh\n")
        f.write("  Camada de saída  : tanh\n\n")
        f.write("[Inicialização dos pesos]\n")
        f.write("  Distribuição uniforme no intervalo [-1, 1]\n\n")
        f.write(f"[Dimensões das matrizes de peso]\n")
        f.write(f"  V (entrada → escondida) : {shape_v}  "
                f"(n_hidden x (n_inputs + 1))\n")
        f.write(f"  W (escondida → saída)   : {shape_w}  "
                f"(n_outputs x (n_hidden + 1))\n")
    print(f"[OK] Hiperparâmetros salvos em: {caminho_hiper}")


def salvar_pesos_iniciais(pasta_saida, V_inicial, W_inicial, map_letras):
    """Gera o arquivo 2: Pesos iniciais (antes do treino)"""
    caminho_pesos_ini = os.path.join(pasta_saida, "pesos_iniciais.txt")
    with open(caminho_pesos_ini, "w", encoding="utf-8") as f:
        f.write("===== PESOS INICIAIS DA REDE (antes do treinamento) =====\n\n")

        f.write(f"--- Matriz V: entrada → camada escondida  {V_inicial.shape} ---\n")
        f.write("# Cada linha é um neurônio escondido; "
                "coluna 0 é o bias, demais são pesos das entradas.\n")
        for j, linha in enumerate(V_inicial):
            f.write(f"  Neurônio escondido {j:3d}: "
                    + "  ".join(f"{v:+.6f}" for v in linha) + "\n")

        f.write(f"\n--- Matriz W: camada escondida → saída  {W_inicial.shape} ---\n")
        f.write("# Cada linha é um neurônio de saída; "
                "coluna 0 é o bias, demais são pesos dos neurônios escondidos.\n")
        for k, linha in enumerate(W_inicial):
            f.write(f"  Neurônio saída {k:2d} ({map_letras[k]}): "
                    + "  ".join(f"{v:+.6f}" for v in linha) + "\n")
    print(f"[OK] Pesos iniciais salvos em: {caminho_pesos_ini}")


def salvar_pesos_finais(pasta_saida, V_final, W_final, historico, menor_erro_val, map_letras):
    """Gera o arquivo 3: Pesos finais (após o treino)"""
    caminho_pesos_fin = os.path.join(pasta_saida, "pesos_finais.txt")
    with open(caminho_pesos_fin, "w", encoding="utf-8") as f:
        f.write("===== PESOS FINAIS DA REDE (após o treinamento) =====\n\n")
        f.write(f"Épocas executadas: {len(historico)}\n")
        f.write(f"Menor erro de validação atingido: {menor_erro_val:.6f}\n\n")

        f.write(f"--- Matriz V: entrada → camada escondida  {V_final.shape} ---\n")
        f.write("# Cada linha é um neurônio escondido; "
                "coluna 0 é o bias, demais são pesos das entradas.\n")
        for j, linha in enumerate(V_final):
            f.write(f"  Neurônio escondido {j:3d}: "
                    + "  ".join(f"{v:+.6f}" for v in linha) + "\n")

        f.write(f"\n--- Matriz W: camada escondida → saída  {W_final.shape} ---\n")
        f.write("# Cada linha é um neurônio de saída; "
                "coluna 0 é o bias, demais são pesos dos neurônios escondidos.\n")
        for k, linha in enumerate(W_final):
            f.write(f"  Neurônio saída {k:2d} ({map_letras[k]}): "
                    + "  ".join(f"{v:+.6f}" for v in linha) + "\n")
    print(f"[OK] Pesos finais salvos em: {caminho_pesos_fin}")


def salvar_historico_erro(pasta_saida, historico):
    """Gera o arquivo 4: Histórico de erro por época"""
    caminho_erro = os.path.join(pasta_saida, "historico_erro.csv")
    with open(caminho_erro, "w", encoding="utf-8") as f:
        f.write("epoca,erro_medio\n")
        for epoca, erro in enumerate(historico, start=1):
            f.write(f"{epoca},{erro:.8f}\n")
    print(f"[OK] Histórico de erro salvo em: {caminho_erro}")


def salvar_saidas_teste(pasta_saida, rede, X_test, T_test, map_letras, n_outputs):
    """Gera o arquivo 5: Saídas produzidas no conjunto de teste e retorna as métricas"""
    acertos = 0
    total = len(X_test)
    caminho_saidas = os.path.join(pasta_saida, "saidas_teste.csv")
    
    with open(caminho_saidas, "w", encoding="utf-8") as f:
        neuronios_header = ",".join(f"y_{map_letras[k]}" for k in range(n_outputs))
        f.write(f"indice,esperado,previsto,acertou,{neuronios_header}\n")

        for i in range(total):
            x = X_test[i]
            t = T_test[i]
            y = rede.predict(x)

            classe_esperada = map_letras[int(np.argmax(t))]
            classe_prevista = map_letras[int(np.argmax(y))]
            acertou = int(classe_esperada == classe_prevista)

            if acertou:
                acertos += 1

            saidas_str = ",".join(f"{v:.6f}" for v in y)
            f.write(f"{i},{classe_esperada},{classe_prevista},{acertou},{saidas_str}\n")

    acuracia = (acertos / total) * 100
    print(f"[OK] Saídas de teste salvas em: {caminho_saidas}")
    
    return acertos, total, acuracia