import pandas as pd
import os

def carregar_dados(caminho_arquivo: str) -> pd.DataFrame:
    """Carrega os dados brutos de análise."""
    print(f"Carregando dados de: {caminho_arquivo}\n")
    return pd.read_csv(caminho_arquivo)

def diagnosticar_dados_por_estado(df: pd.DataFrame, estado: str):
    """
    Realiza uma análise diagnóstica para um estado específico,
    identificando meses com dados zerados ou suspeitos.
    """
    print("="*50)
    print(f"🔍 DIAGNÓSTICO PARA O ESTADO: {estado.upper()}")
    print("="*50)
    
    df_estado = df[df['uf'] == estado].copy()
    
    if df_estado.empty:
        print("Nenhum dado encontrado para este estado.")
        return
        
    # Agrupa por ano e mês para análise
    analise_mensal = df_estado.groupby(['ano', 'mes']).agg(
        total_homicidios=('homicidio_doloso', 'sum'),
        municipios_reportados=('id_municipio', 'nunique')
    ).reset_index()
    
    # Identifica meses com zero homicídios
    meses_zerados = analise_mensal[analise_mensal['total_homicidios'] == 0]
    
    if not meses_zerados.empty:
        print("\n❗️ ALERTA: Encontrados meses com registro de ZERO homicídios:")
        print(meses_zerados.to_string(index=False))
    else:
        print("\n✅ Nenhum mês com total de homicídios zerado foi encontrado.")
        
    # Exibe um resumo da contagem de meses com dados por ano
    print("\n🗓️ Resumo de meses com dados disponíveis por ano:")
    contagem_anual = df_estado.groupby('ano')['mes'].nunique().reset_index()
    contagem_anual.rename(columns={'mes': 'meses_com_dados'}, inplace=True)
    print(contagem_anual.to_string(index=False))
    print("\n")


if __name__ == "__main__":
    PASTA_DADOS_TRATADOS = 'dados_tratados'
    ARQUIVO_ENTRADA = 'analise_seguranca_sp_rj.csv'
    caminho_arquivo = os.path.join(PASTA_DADOS_TRATADOS, ARQUIVO_ENTRADA)

    try:
        dados_completos = carregar_dados(caminho_arquivo)
        
        # Executa o diagnóstico para cada estado
        diagnosticar_dados_por_estado(dados_completos, 'SP')
        diagnosticar_dados_por_estado(dados_completos, 'RJ')
        
        print("="*50)
        print("Diagnóstico Concluído.")
        print("="*50)

    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado em {caminho_arquivo}")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")