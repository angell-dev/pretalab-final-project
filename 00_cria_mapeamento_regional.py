import pandas as pd
import os

def ler_municipios_ibge(caminho_ibge: str) -> pd.DataFrame:
    """
    Lê o arquivo do IBGE, filtra para SP e RJ, e extrai as informações
    relevantes de municípios e regiões geográficas.
    """
    print(f"1. Lendo arquivo de municípios do IBGE: {caminho_ibge}")
    
    # Usar engine='xlrd' para garantir a leitura de arquivos .xls legados
    df_ibge = pd.read_excel(caminho_ibge, engine='xlrd')
    
    # Selecionar e renomear colunas de interesse
    colunas = {
        'UF': 'uf',
        'Nome_UF': 'nome_uf',
        'Nome Região Geográfica Intermediária': 'regiao',
        'Código Município Completo': 'id_municipio_ibge',
        'Nome_Município': 'nome_municipio'
    }
    df_ibge = df_ibge[list(colunas.keys())].rename(columns=colunas)

    # Filtrar apenas para SP e RJ
    df_sp_rj = df_ibge[df_ibge['uf'].isin(['SP', 'RJ'])].copy()
    
    # O ID do município na Base dos Dados é o código do IBGE com 6 dígitos.
    # O código do IBGE tem 7 dígitos. Precisamos remover o último (dígito verificador).
    df_sp_rj['id_municipio'] = df_sp_rj['id_municipio_ibge'].astype(str).str[:6].astype(int)
    
    print("   - Leitura do arquivo IBGE concluída.")
    return df_sp_rj[['id_municipio', 'nome_municipio', 'uf', 'regiao']]

def ler_regioes_rj_isp(caminho_isp: str) -> pd.DataFrame:
    """
    Lê o arquivo do ISP-RJ para extrair o mapeamento oficial das regiões de segurança.
    """
    print(f"2. Lendo arquivo de regiões do ISP-RJ: {caminho_isp}")
    df_isp = pd.read_csv(caminho_isp)
    
    # O id_municipio aqui já tem 6 dígitos, o que é ótimo.
    df_regioes_rj = df_isp[['id_municipio', 'regiao']].drop_duplicates().copy()
    
    # Renomear para evitar conflito na hora de juntar
    df_regioes_rj.rename(columns={'regiao': 'regiao_isp'}, inplace=True)
    
    print("   - Leitura das regiões do RJ concluída.")
    return df_regioes_rj

def criar_mapeamento_final(df_ibge_sp_rj: pd.DataFrame, df_regioes_rj: pd.DataFrame, caminho_saida: str):
    """
    Unifica os dados de SP e RJ, usando as regiões do ISP para o RJ,
    e salva o mapeamento final em um arquivo CSV.
    """
    print("3. Unificando mapeamentos de SP e RJ...")
    
    # Separar SP
    df_sp = df_ibge_sp_rj[df_ibge_sp_rj['uf'] == 'SP'].copy()
    
    # Para o RJ, vamos usar as regiões do ISP, que são mais específicas para segurança.
    df_rj_ibge = df_ibge_sp_rj[df_ibge_sp_rj['uf'] == 'RJ'].copy()
    
    # Juntar os dados do RJ (IBGE) com as regiões do RJ (ISP)
    df_rj_final = pd.merge(df_rj_ibge, df_regioes_rj, on='id_municipio', how='left')
    
    # Usar a região do ISP como a definitiva para o RJ e descartar a do IBGE
    df_rj_final['regiao'] = df_rj_final['regiao_isp']
    df_rj_final.drop(columns=['regiao_isp'], inplace=True)
    
    # Combinar os dataframes de SP e RJ de volta em um só
    df_mapeamento = pd.concat([df_sp, df_rj_final], ignore_index=True)
    
    # Salvar o arquivo final na pasta de dados tratados
    df_mapeamento.to_csv(caminho_saida, index=False)
    print(f"\n✅ Mapeamento final salvo com sucesso em: {caminho_saida}")
    print("\nAmostra do mapeamento criado:")
    print(df_mapeamento.sample(10))
    

if __name__ == "__main__":
    print("="*60)
    print("INICIANDO CRIAÇÃO DO MAPEAMENTO DE MUNICÍPIOS E REGIÕES")
    print("="*60)

    # Caminhos para os arquivos de dados brutos
    PASTA_DADOS_BRUTOS = 'dados_brutos'
    PASTA_DADOS_TRATADOS = 'dados_tratados'
    
    # Garante que a pasta de saída exista
    if not os.path.exists(PASTA_DADOS_TRATADOS):
        os.makedirs(PASTA_DADOS_TRATADOS)
        
    CAMINHO_IBGE_XLS = os.path.join(PASTA_DADOS_BRUTOS, 'RELATORIO_DTB_BRASIL_2024_MUNICIPIOS.xls')
    CAMINHO_ISP_CSV = os.path.join(PASTA_DADOS_BRUTOS, 'seguranca_publica', 'br_rj_isp_estatisticas_seguranca_evolucao_mensal_municipio.csv')
    CAMINHO_SAIDA_CSV = os.path.join(PASTA_DADOS_TRATADOS, 'mapeamento_municipio_regiao.csv')

    try:
        df_ibge = ler_municipios_ibge(CAMINHO_IBGE_XLS)
        df_regioes_rj_isp = ler_regioes_rj_isp(CAMINHO_ISP_CSV)
        criar_mapeamento_final(df_ibge, df_regioes_rj_isp, CAMINHO_SAIDA_CSV)
        
        print("\n" + "="*60)
        print("PROCESSO CONCLUÍDO COM SUCESSO!")
        print("="*60)

    except FileNotFoundError as e:
        print(f"\n❌ ERRO: Arquivo não encontrado. Verifique o caminho: {e.filename}")
    except Exception as e:
        print(f"\n❌ Ocorreu um erro inesperado: {e}")