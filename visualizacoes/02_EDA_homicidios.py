import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

def carregar_dados(caminho_arquivo: str) -> pd.DataFrame:
    """Carrega os dados de análise, garantindo o parse da data."""
    print(f"1. Carregando dados de: {caminho_arquivo}")
    return pd.read_csv(caminho_arquivo, parse_dates=['data'])

def preparar_dados_homicidios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara os dados para a análise, removendo cirurgicamente os dados
    sabidamente errados e encontrando o período comum.
    """
    print("2. Preparando e limpando dados de homicídios...")
    
    print("Removendo dados comprovadamente errados (Homicídios zerados em SP no final de 2021)...")
    condicao_erro = (df['uf'] == 'SP') & (df['ano'] == 2021) & (df['mes'].isin([9, 10, 11, 12]))
    df_corrigido = df[~condicao_erro]
    
    # O resto do processo continua como antes, mas agora sobre dados limpos.
    df_pivot = df_corrigido.pivot_table(
        index='data', 
        columns='uf', 
        values='homicidio_doloso',
        aggfunc='sum'
    )
    
    df_resampled = df_pivot.resample('MS').sum(min_count=1)
    df_final = df_resampled.dropna()
    
    df_final = df_final.rename(columns={'RJ': 'Rio de Janeiro', 'SP': 'São Paulo'})
    
    print("\nPeríodo de análise final, contínuo e comum encontrado:")
    print(f"Data de início: {df_final.index.min().strftime('%Y-%m')}")
    print(f"Data de fim:    {df_final.index.max().strftime('%Y-%m')}")
    
    return df_final

def gerar_analise_descritiva(df: pd.DataFrame):
    """Exibe um resumo estatístico formatado dos dados."""
    print("\n3. Análise Estatística Descritiva (Dados Finais):")
    summary = df.describe().T
    format_dict = {
        'count': '{:.0f}', 'mean': '{:.2f}', 'std': '{:.2f}',
        'min': '{:.0f}', '25%': '{:.1f}', '50%': '{:.1f}',
        '75%': '{:.1f}', 'max': '{:.0f}'
    }
    # Adicionado try-except para o caso de o Jinja2 não estar instalado
    try:
        print(summary.style.format(format_dict).to_string())
    except ImportError:
        print(summary)

# As funções de plotagem permanecem as mesmas do script anterior
def plotar_serie_temporal(df: pd.DataFrame, pasta_saida: str):
    """Plota a evolução mensal dos homicídios com eixo X formatado."""
    caminho_grafico = os.path.join(pasta_saida, '01_homicidios_evolucao_mensal_FINAL.png')
    print(f"4. Gerando Gráfico 1: Série Temporal em {caminho_grafico}")
    
    plt.figure(figsize=(15, 7))
    ax = sns.lineplot(data=df, dashes=False, linewidth=2.5)
    
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=0, ha='center')
    
    plt.title('Evolução Mensal de Homicídios Dolosos (Dados Validados)', fontsize=16, weight='bold')
    plt.xlabel('Ano', fontsize=12)
    plt.ylabel('Número de Homicídios', fontsize=12)
    plt.legend(title='Estado')
    plt.tight_layout()
    plt.savefig(caminho_grafico)
    plt.close()

def plotar_comparativo_anual(df: pd.DataFrame, pasta_saida: str):
    """Plota um gráfico de barras com o total de homicídios por ano."""
    caminho_grafico = os.path.join(pasta_saida, '02_homicidios_comparativo_anual_FINAL.png')
    print(f"5. Gerando Gráfico 2: Comparativo Anual em {caminho_grafico}")
    
    df_anual = df.resample('YE').sum()
    df_anual.index = df_anual.index.year
    
    df_anual.plot(kind='bar', figsize=(15, 7), width=0.8)
    
    plt.title('Total Anual de Homicídios Dolosos (SP vs. RJ)', fontsize=16, weight='bold')
    plt.xlabel('Ano', fontsize=12)
    plt.ylabel('Número Total de Homicídios', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Estado')
    plt.tight_layout()
    plt.savefig(caminho_grafico)
    plt.close()

def plotar_distribuicao_mensal(df: pd.DataFrame, pasta_saida: str):
    """Cria um boxplot para analisar a distribuição por mês do ano."""
    caminho_grafico = os.path.join(pasta_saida, '03_homicidios_distribuicao_mensal_FINAL.png')
    print(f"6. Gerando Gráfico 3: Distribuição Mensal (Sazonalidade) em {caminho_grafico}")
    
    df_melted = df.copy()
    df_melted['Mês'] = df_melted.index.strftime('%m - %b')
    df_melted = df_melted.melt(id_vars='Mês', var_name='Estado', value_name='Homicídios')
    
    plt.figure(figsize=(15, 7))
    sns.boxplot(data=df_melted.sort_values('Mês'), x='Mês', y='Homicídios', hue='Estado')
    
    plt.title('Distribuição Mensal de Homicídios (Sazonalidade)', fontsize=16, weight='bold')
    plt.xlabel('Mês do Ano', fontsize=12)
    plt.ylabel('Número de Homicídios', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Estado')
    plt.tight_layout()
    plt.savefig(caminho_grafico)
    plt.close()


if __name__ == "__main__":
    PASTA_DADOS_TRATADOS = 'dados_tratados'
    PASTA_VISUALIZACOES = 'visualizacoes'
    ARQUIVO_ENTRADA = 'analise_seguranca_sp_rj.csv'

    if not os.path.exists(PASTA_VISUALIZACOES):
        os.makedirs(PASTA_VISUALIZACOES)

    caminho_arquivo_entrada = os.path.join(PASTA_DADOS_TRATADOS, ARQUIVO_ENTRADA)

    try:
        dados_brutos = carregar_dados(caminho_arquivo_entrada)
        dados_homicidios_comum = preparar_dados_homicidios(dados_brutos)
        
        if dados_homicidios_comum.empty:
            print("\nAVISO: Nenhum período com dados contínuos para ambos os estados foi encontrado.")
        else:
            gerar_analise_descritiva(dados_homicidios_comum)
            plotar_serie_temporal(dados_homicidios_comum, PASTA_VISUALIZACOES)
            plotar_comparativo_anual(dados_homicidios_comum, PASTA_VISUALIZACOES)
            plotar_distribuicao_mensal(dados_homicidios_comum, PASTA_VISUALIZACOES)

            print("\n--- Processo de EDA Concluído ---")
            print("Gráficos FINAIS foram salvos na pasta 'visualizacoes'.")

    except Exception as e:
        print(f"\nOcorreu um erro inesperado durante a execução: {e}")
