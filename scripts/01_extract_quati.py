import pandas as pd
from datasets import load_dataset
import os


# 1.(Parâmetros)

dataset = "unicamp-dl/quati"
nome_da_tabela = "quati_1M_passages"
caminho_pasta = "data/raw/"
nome_do_arquivo = "quati_1m_sample.csv"
QUANTIDADE_LINHAS = 10000


# 2. Processamento

def baixar_dados_brutos():
    print("Iniciando o download do dataset Quati (Unicamp)...")
    dados_brutos = load_dataset(dataset, nome_da_tabela, split=nome_da_tabela, trust_remote_code=True)
    tabela_dados = pd.DataFrame(dados_brutos)
    return tabela_dados

def salvar_dados(tabela):
    print("Preparando para salvar os dados...")
    os.makedirs(caminho_pasta, exist_ok=True)
    caminho_completo = os.path.join(caminho_pasta, nome_do_arquivo)
    amostra_tabela = tabela.head(QUANTIDADE_LINHAS)
    amostra_tabela.to_csv(caminho_completo, index=False)
    print(f"Sucesso! Arquivo salvo em: {caminho_completo}")


# 3. Execução

if __name__ == "__main__":
    tabela_quati = baixar_dados_brutos()
    salvar_dados(tabela_quati)