import pandas as pd
import os

# --- Funções Auxiliares ---
# É uma boa prática organizar o código em funções para clareza e reutilização.

def carregar_dados_consolidados(caminho_entrada: str) -> pd.DataFrame:
    """
    Carrega o arquivo CSV consolidado da segurança pública.

    Args:
        caminho_entrada (str): O caminho completo para o arquivo CSV.

    Returns:
        pd.DataFrame: O DataFrame carregado com os dados.
    """
    print(f"Carregando dados de: {caminho_entrada}")
    if not os.path.exists(caminho_entrada):
        raise FileNotFoundError(f"Arquivo não encontrado em: {caminho_entrada}")
    
    # Tenta ler o arquivo com o encoding padrão (utf-8), se falhar, tenta o latin1.
    try:
        df = pd.read_csv(caminho_entrada)
    except UnicodeDecodeError:
        df = pd.read_csv(caminho_entrada, encoding='latin1')
        
    print("Dados carregados com sucesso.")
    return df

def limpar_e_preparar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza a limpeza e transformação inicial do DataFrame.

    Args:
        df (pd.DataFrame): O DataFrame a ser limpo.

    Returns:
        pd.DataFrame: O DataFrame limpo e preparado.
    """
    print("Iniciando limpeza e preparação dos dados...")
    
    # Padroniza nomes das colunas para minúsculas para facilitar o acesso.
    df.columns = df.columns.str.lower()
    
    # Cria uma coluna de data real a partir do ano e mês.
    # O dia é definido como 1 para criar o objeto datetime.
    print("Criando coluna 'data' a partir de 'ano' e 'mes'...")
    df['data'] = pd.to_datetime(df['ano'].astype(str) + '-' + df['mes'].astype(str) + '-01')
    
    # Verifica a quantidade de valores nulos por coluna.
    print("\nVerificação de valores nulos por coluna:")
    print(df.isnull().sum())
    
    # Vamos preencher valores nulos em colunas de crimes com 0.
    # Isso assume que um valor nulo significa 'nenhuma ocorrência registrada'.
    # É uma premissa importante que devemos validar depois, mas é um bom começo.
    colunas_crimes = ['homicidio_doloso', 'roubo_veiculo', 'furto_veiculo', 'roubo_total', 'furto_total']
    df[colunas_crimes] = df[colunas_crimes].fillna(0)
    print("\nValores nulos nas colunas de crimes preenchidos com 0.")
    
    # Reorganiza as colunas para melhor visualização.
    colunas_ordem = ['data', 'ano', 'mes', 'uf', 'id_municipio'] + colunas_crimes
    df = df[colunas_ordem]
    
    print("\nLimpeza concluída. Amostra dos dados transformados:")
    print(df.head())
    print("\nTipos de dados após a conversão:")
    df.info(memory_usage=False)
    
    return df

def salvar_dados_tratados(df: pd.DataFrame, caminho_saida: str):
    """
    Salva o DataFrame limpo em um novo arquivo CSV.

    Args:
        df (pd.DataFrame): O DataFrame a ser salvo.
        caminho_saida (str): O caminho para o novo arquivo CSV.
    """
    print(f"\nSalvando arquivo tratado em: {caminho_saida}")
    df.to_csv(caminho_saida, index=False)
    print("Arquivo salvo com sucesso!")


# --- Bloco Principal de Execução ---
# Este bloco é executado quando o script é chamado diretamente.
if __name__ == "__main__":
    # Define os caminhos de forma relativa para funcionar em qualquer máquina.
    PASTA_DADOS_TRATADOS = 'dados_tratados'
    ARQUIVO_ENTRADA = 'seguranca_publica_sp_rj_consolidado.csv'
    ARQUIVO_SAIDA = 'analise_seguranca_sp_rj.csv'

    caminho_completo_entrada = os.path.join(PASTA_DADOS_TRATADOS, ARQUIVO_ENTRADA)
    caminho_completo_saida = os.path.join(PASTA_DADOS_TRATADOS, ARQUIVO_SAIDA)

    # Executa o pipeline: carregar -> limpar -> salvar
    try:
        dados_brutos = carregar_dados_consolidados(caminho_completo_entrada)
        dados_limpos = limpar_e_preparar_dados(dados_brutos)
        salvar_dados_tratados(dados_limpos, caminho_completo_saida)

        print("\n--- Processo Concluído ---")
        print("O arquivo 'analise_seguranca_sp_rj.csv' está pronto para a próxima fase.")

    except FileNotFoundError as e:
        print(f"\nERRO: {e}")
        print("Verifique se o nome do arquivo de entrada e o caminho estão corretos.")
    except Exception as e:
        print(f"\nOcorreu um erro inesperado: {e}")