import pandas as pd
import numpy as np
import os
from scipy import stats
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import ruptures as rpt
import warnings
warnings.filterwarnings('ignore')

def carregar_dados_preparados(caminho_arquivo: str) -> pd.DataFrame:
    """
    Carrega os dados já tratados e remove o período com erro conhecido.
    """
    print("=" * 60)
    print("ANÁLISE DE SIGNIFICÂNCIA ESTATÍSTICA - HOMICÍDIOS SP vs RJ")
    print("=" * 60)
    
    df = pd.read_csv(caminho_arquivo, parse_dates=['data'])
    
    # Remove período com erro conhecido em SP
    condicao_erro = (df['uf'] == 'SP') & (df['ano'] == 2021) & (df['mes'].isin([9, 10, 11, 12]))
    df = df[~condicao_erro]
    
    return df

def teste_hipoteses_comparativo(df: pd.DataFrame):
    """
    Realiza testes estatísticos para comparar as distribuições de homicídios entre SP e RJ.
    """
    print("\n1. TESTES DE HIPÓTESES COMPARATIVOS")
    print("-" * 40)
    
    # Preparar dados por estado
    sp_data = df[df['uf'] == 'SP']['homicidio_doloso'].values
    rj_data = df[df['uf'] == 'RJ']['homicidio_doloso'].values
    
    # 1. Teste de Normalidade (Shapiro-Wilk)
    _, p_sp_normal = stats.shapiro(sp_data)
    _, p_rj_normal = stats.shapiro(rj_data)
    
    print(f"\nTeste de Normalidade (Shapiro-Wilk):")
    print(f"  São Paulo: p-valor = {p_sp_normal:.4f} {'(Normal)' if p_sp_normal > 0.05 else '(Não-Normal)'}")
    print(f"  Rio de Janeiro: p-valor = {p_rj_normal:.4f} {'(Normal)' if p_rj_normal > 0.05 else '(Não-Normal)'}")
    
    # 2. Teste de Variâncias (Levene)
    _, p_levene = stats.levene(sp_data, rj_data)
    print(f"\nTeste de Homogeneidade de Variâncias (Levene):")
    print(f"  p-valor = {p_levene:.4f} {'(Variâncias iguais)' if p_levene > 0.05 else '(Variâncias diferentes)'}")
    
    # 3. Teste de Médias (t-test ou Mann-Whitney baseado na normalidade)
    if p_sp_normal > 0.05 and p_rj_normal > 0.05:
        t_stat, p_ttest = stats.ttest_ind(sp_data, rj_data, equal_var=(p_levene > 0.05))
        print(f"\nTeste t de Student:")
        print(f"  Estatística t = {t_stat:.4f}")
        print(f"  p-valor = {p_ttest:.4f}")
    else:
        u_stat, p_mann = stats.mannwhitneyu(sp_data, rj_data, alternative='two-sided')
        print(f"\nTeste de Mann-Whitney U:")
        print(f"  Estatística U = {u_stat:.4f}")
        print(f"  p-valor = {p_mann:.4f}")
    
    # 4. Tamanho do efeito (Cohen's d)
    cohens_d = (np.mean(rj_data) - np.mean(sp_data)) / np.sqrt((np.std(sp_data)**2 + np.std(rj_data)**2) / 2)
    print(f"\nTamanho do Efeito (Cohen's d): {cohens_d:.4f}")
    print(f"  Interpretação: {'Pequeno' if abs(cohens_d) < 0.5 else 'Médio' if abs(cohens_d) < 0.8 else 'Grande'}")
    
    return {
        'normalidade': {'sp': p_sp_normal > 0.05, 'rj': p_rj_normal > 0.05},
        'variancias_iguais': p_levene > 0.05,
        'diferenca_significativa': p_ttest < 0.05 if p_sp_normal > 0.05 and p_rj_normal > 0.05 else p_mann < 0.05,
        'tamanho_efeito': cohens_d
    }

def analise_changepoint(df: pd.DataFrame, pasta_saida: str):
    """
    Detecta pontos de mudança estrutural nas séries temporais de homicídios.
    """
    print("\n2. ANÁLISE DE PONTOS DE MUDANÇA (CHANGE POINTS)")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    for idx, estado in enumerate(['SP', 'RJ']):
        # Filtrar dados do estado
        df_estado = df[df['uf'] == estado].copy()
        df_estado = df_estado.sort_values('data')
        serie = df_estado.groupby('data')['homicidio_doloso'].sum().values
        
        # Detecção de changepoints usando o algoritmo PELT
        algo = rpt.Pelt(model="rbf").fit(serie)
        result = algo.predict(pen=10)
        
        # Plotar série com changepoints
        ax = axes[idx]
        ax.plot(serie, label=f'Homicídios - {estado}', color='darkblue' if estado == 'SP' else 'darkred')
        
        # Marcar changepoints
        for cp in result[:-1]:  # Excluir o último ponto (fim da série)
            ax.axvline(x=cp, color='red', linestyle='--', alpha=0.7)
            
        ax.set_title(f'Detecção de Mudanças Estruturais - {estado}', fontsize=14, weight='bold')
        ax.set_xlabel('Meses desde o início da série')
        ax.set_ylabel('Número de Homicídios')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Imprimir datas dos changepoints
        datas_serie = df_estado.groupby('data')['homicidio_doloso'].sum().index
        print(f"\n{estado} - Pontos de mudança detectados:")
        for cp in result[:-1]:
            if cp < len(datas_serie):
                data_mudanca = datas_serie[cp]
                print(f"  - {data_mudanca.strftime('%Y-%m')}")
    
    plt.tight_layout()
    caminho_grafico = os.path.join(pasta_saida, '04_changepoints_analysis.png')
    plt.savefig(caminho_grafico, dpi=300)
    plt.close()
    
    return result

def decomposicao_sazonal_avancada(df: pd.DataFrame, pasta_saida: str):
    """
    Realiza decomposição STL (Seasonal and Trend decomposition using Loess) avançada.
    """
    print("\n3. DECOMPOSIÇÃO SAZONAL AVANÇADA (STL)")
    print("-" * 40)
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    
    resultados_stl = {}
    
    for idx, estado in enumerate(['SP', 'RJ']):
        # Preparar série temporal
        df_estado = df[df['uf'] == estado].copy()
        serie_mensal = df_estado.groupby('data')['homicidio_doloso'].sum()
        serie_mensal = serie_mensal.asfreq('MS')  # Garantir frequência mensal
        
        # Decomposição STL
        stl = STL(serie_mensal, seasonal=13)  # seasonal=13 para capturar padrões anuais
        resultado = stl.fit()
        
        resultados_stl[estado] = resultado
        
        # Plotar componentes
        componentes = ['Série Original', 'Tendência', 'Sazonalidade', 'Resíduo']
        dados_plot = [serie_mensal, resultado.trend, resultado.seasonal, resultado.resid]
        
        for i, (componente, dados) in enumerate(zip(componentes, dados_plot)):
            ax = axes[i, idx]
            ax.plot(dados, color='darkblue' if estado == 'SP' else 'darkred')
            ax.set_title(f'{componente} - {estado}', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            if i == 0:  # Série original
                ax.set_ylabel('Homicídios')
            elif i == 1:  # Tendência
                ax.set_ylabel('Tendência')
            elif i == 2:  # Sazonalidade
                ax.set_ylabel('Componente Sazonal')
                # Destacar força da sazonalidade
                forca_sazonal = np.std(resultado.seasonal) / np.std(serie_mensal)
                ax.text(0.02, 0.95, f'Força: {forca_sazonal:.2%}', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:  # Resíduo
                ax.set_ylabel('Resíduo')
    
    plt.tight_layout()
    caminho_grafico = os.path.join(pasta_saida, '05_decomposicao_stl.png')
    plt.savefig(caminho_grafico, dpi=300)
    plt.close()
    
    # Análise da força sazonal
    print("\nForça da Componente Sazonal:")
    for estado in ['SP', 'RJ']:
        serie_mensal = df[df['uf'] == estado].groupby('data')['homicidio_doloso'].sum()
        forca = np.std(resultados_stl[estado].seasonal) / np.std(serie_mensal)
        print(f"  {estado}: {forca:.2%}")
    
    return resultados_stl

def analise_tendencia_estatistica(df: pd.DataFrame):
    """
    Análise estatística das tendências temporais usando regressão.
    """
    print("\n4. ANÁLISE DE TENDÊNCIAS TEMPORAIS")
    print("-" * 40)
    
    resultados_tendencia = {}
    
    for estado in ['SP', 'RJ']:
        df_estado = df[df['uf'] == estado].copy()
        serie_mensal = df_estado.groupby('data')['homicidio_doloso'].sum().reset_index()
        
        # Criar variável temporal numérica (meses desde início)
        serie_mensal['tempo'] = range(len(serie_mensal))
        
        # Regressão linear para tendência
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            serie_mensal['tempo'], 
            serie_mensal['homicidio_doloso']
        )
        
        # Calcular taxa de mudança mensal e anual
        taxa_mudanca_mensal = slope / serie_mensal['homicidio_doloso'].mean() * 100
        taxa_mudanca_anual = taxa_mudanca_mensal * 12
        
        resultados_tendencia[estado] = {
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'taxa_mudanca_mensal': taxa_mudanca_mensal,
            'taxa_mudanca_anual': taxa_mudanca_anual
        }
        
        print(f"\n{estado}:")
        print(f"  Coeficiente angular: {slope:.2f} homicídios/mês")
        print(f"  R² = {r_value**2:.4f}")
        print(f"  p-valor = {p_value:.4f} {'(Significativo)' if p_value < 0.05 else '(Não significativo)'}")
        print(f"  Taxa de mudança: {taxa_mudanca_mensal:.2f}% ao mês ({taxa_mudanca_anual:.1f}% ao ano)")
    
    return resultados_tendencia

def criar_dashboard_estatistico(resultados: dict, pasta_saida: str):
    """
    Cria um dashboard visual com os principais resultados estatísticos.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Dashboard de Análise Estatística - Homicídios SP vs RJ', fontsize=16, weight='bold')
    
    # Definir grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Comparação de médias e intervalos de confiança
    ax1 = fig.add_subplot(gs[0, :2])
    estados = ['SP', 'RJ']
    medias = resultados['estatisticas_descritivas']['medias']
    erros = resultados['estatisticas_descritivas']['erros_padrao']
    
    bars = ax1.bar(estados, medias, yerr=erros, capsize=10, 
                    color=['#1f77b4', '#ff7f0e'], alpha=0.8)
    ax1.set_ylabel('Média de Homicídios Mensais')
    ax1.set_title('Comparação de Médias com IC 95%')
    
    # Adicionar valores nas barras
    for bar, media in zip(bars, medias):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{media:.0f}', ha='center', va='bottom')
    
    # 2. Tamanho do efeito
    ax2 = fig.add_subplot(gs[0, 2])
    cohens_d = resultados['testes']['tamanho_efeito']
    color = 'green' if abs(cohens_d) < 0.5 else 'orange' if abs(cohens_d) < 0.8 else 'red'
    
    ax2.barh(['Effect Size'], [abs(cohens_d)], color=color, alpha=0.8)
    ax2.set_xlim(0, 2)
    ax2.set_xlabel("Cohen's d")
    ax2.set_title('Tamanho do Efeito')
    ax2.text(abs(cohens_d) + 0.1, 0, f'{cohens_d:.2f}', va='center')
    
    # 3. Tendências temporais
    ax3 = fig.add_subplot(gs[1, :])
    x = np.arange(2)
    width = 0.35
    
    taxas_anuais = [resultados['tendencias'][estado]['taxa_mudanca_anual'] for estado in estados]
    p_valores = [resultados['tendencias'][estado]['p_value'] for estado in estados]
    
    bars = ax3.bar(x, taxas_anuais, width, 
                    color=['#1f77b4', '#ff7f0e'], alpha=0.8)
    
    ax3.set_ylabel('Taxa de Mudança Anual (%)')
    ax3.set_title('Tendências Temporais')
    ax3.set_xticks(x)
    ax3.set_xticklabels(estados)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Adicionar significância
    for i, (bar, p_val) in enumerate(zip(bars, p_valores)):
        height = bar.get_height()
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                sig, ha='center', va='bottom' if height > 0 else 'top')
    
    # 4. Resumo estatístico em texto
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    texto_resumo = f"""
    RESUMO DOS RESULTADOS:
    
    • Diferença nas médias: {'Significativa' if resultados['testes']['diferenca_significativa'] else 'Não significativa'} 
    (p < 0.05)
    
    • RJ tem em média {(medias[1]/medias[0] - 1)*100:.1f}% mais homicídios que SP
    
    • Tendências: SP está {'aumentando' if taxas_anuais[0] > 0 else 'diminuindo'} {abs(taxas_anuais[0]):.1f}% ao ano
                RJ está {'aumentando' if taxas_anuais[1] > 0 else 'diminuindo'} {abs(taxas_anuais[1]):.1f}% ao ano
    
    • Sazonalidade mais forte em: {resultados['estado_maior_sazonalidade']}
    """
    
    ax4.text(0.1, 0.9, texto_resumo, transform=ax4.transAxes, 
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    caminho_dashboard = os.path.join(pasta_saida, '06_dashboard_estatistico.png')
    plt.savefig(caminho_dashboard, dpi=300)
    plt.close()

def gerar_relatorio_csv(resultados: dict, pasta_saida: str):
    """
    Gera um relatório em CSV com todos os resultados estatísticos para uso no Power BI.
    """
    # Criar DataFrame com resultados
    dados_relatorio = []
    
    for estado in ['SP', 'RJ']:
        dados_relatorio.append({
            'Estado': estado,
            'Media_Homicidios': resultados['estatisticas_descritivas']['medias'][0 if estado == 'SP' else 1],
            'Desvio_Padrao': resultados['estatisticas_descritivas']['desvios'][0 if estado == 'SP' else 1],
            'Taxa_Mudanca_Anual_%': resultados['tendencias'][estado]['taxa_mudanca_anual'],
            'Tendencia_Significativa': 'Sim' if resultados['tendencias'][estado]['p_value'] < 0.05 else 'Não',
            'R_Quadrado': resultados['tendencias'][estado]['r_squared'],
            'Forca_Sazonalidade': resultados['forca_sazonalidade'][estado]
        })
    
    df_relatorio = pd.DataFrame(dados_relatorio)
    
    # Adicionar comparações
    df_comparacao = pd.DataFrame([{
        'Metrica': 'Diferenca_Medias_Significativa',
        'Valor': 'Sim' if resultados['testes']['diferenca_significativa'] else 'Não',
        'P_Valor': resultados['testes'].get('p_valor', None),
        'Tamanho_Efeito_Cohens_d': resultados['testes']['tamanho_efeito']
    }])
    
    # Salvar arquivos
    df_relatorio.to_csv(os.path.join(pasta_saida, 'resultados_estatisticos_por_estado.csv'), index=False)
    df_comparacao.to_csv(os.path.join(pasta_saida, 'resultados_comparacao_estados.csv'), index=False)
    
    print("\n5. ARQUIVOS PARA POWER BI GERADOS:")
    print("   - resultados_estatisticos_por_estado.csv")
    print("   - resultados_comparacao_estados.csv")

if __name__ == "__main__":
    PASTA_DADOS_TRATADOS = 'dados_tratados'
    PASTA_VISUALIZACOES = 'visualizacoes'
    ARQUIVO_ENTRADA = 'analise_seguranca_sp_rj.csv'
    
    caminho_arquivo = os.path.join(PASTA_DADOS_TRATADOS, ARQUIVO_ENTRADA)
    
    try:
        # Carregar dados
        dados = carregar_dados_preparados(caminho_arquivo)
        
        # Executar análises
        resultados_testes = teste_hipoteses_comparativo(dados)
        changepoints = analise_changepoint(dados, PASTA_VISUALIZACOES)
        resultados_stl = decomposicao_sazonal_avancada(dados, PASTA_VISUALIZACOES)
        resultados_tendencia = analise_tendencia_estatistica(dados)
        
        # Calcular estatísticas descritivas para o dashboard
        sp_data = dados[dados['uf'] == 'SP']['homicidio_doloso'].values
        rj_data = dados[dados['uf'] == 'RJ']['homicidio_doloso'].values
        
        # Calcular força da sazonalidade
        forca_sazonal = {}
        for estado in ['SP', 'RJ']:
            serie = dados[dados['uf'] == estado].groupby('data')['homicidio_doloso'].sum()
            forca_sazonal[estado] = np.std(resultados_stl[estado].seasonal) / np.std(serie)
        
        # Compilar resultados
        resultados_completos = {
            'testes': resultados_testes,
            'tendencias': resultados_tendencia,
            'estatisticas_descritivas': {
                'medias': [np.mean(sp_data), np.mean(rj_data)],
                'desvios': [np.std(sp_data), np.std(rj_data)],
                'erros_padrao': [stats.sem(sp_data) * 1.96, stats.sem(rj_data) * 1.96]
            },
            'forca_sazonalidade': forca_sazonal,
            'estado_maior_sazonalidade': max(forca_sazonal, key=forca_sazonal.get)
        }
        
        # Gerar dashboard e relatórios
        criar_dashboard_estatistico(resultados_completos, PASTA_VISUALIZACOES)
        gerar_relatorio_csv(resultados_completos, PASTA_DADOS_TRATADOS)
        
        print("\n" + "=" * 60)
        print("ANÁLISE DE SIGNIFICÂNCIA CONCLUÍDA COM SUCESSO!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERRO durante a execução: {e}")
        import traceback
        traceback.print_exc()
