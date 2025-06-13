import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def carregar_dados_municipais(pasta_dados: str) -> pd.DataFrame:
    """
    Carrega e consolida dados municipais de SP e RJ.
    """
    print("=" * 60)
    print("ANÁLISE ESPACIAL E CLUSTERING DE MUNICÍPIOS")
    print("=" * 60)
    
    # Carregar dados gerais
    df_geral = pd.read_csv(os.path.join(pasta_dados, 'analise_seguranca_sp_rj.csv'))
    
    # Vamos focar no período mais recente para análise espacial
    ano_analise = 2021
    df_recente = df_geral[df_geral['ano'] == ano_analise].copy()
    
    # Remover meses com problemas conhecidos
    if 'SP' in df_recente['uf'].values:
        df_recente = df_recente[~((df_recente['uf'] == 'SP') & (df_recente['mes'].isin([9, 10, 11, 12])))]
    
    print(f"\nAnalisando dados de {ano_analise}")
    print(f"Total de registros: {len(df_recente)}")
    print(f"Municípios únicos: {df_recente['id_municipio'].nunique()}")
    
    return df_recente, df_geral

def criar_matriz_crimes_municipios(df_recente: pd.DataFrame, df_geral: pd.DataFrame) -> pd.DataFrame:
    """
    Cria matriz consolidada de crimes por município com métricas calculadas.
    """
    print("\n1. CRIANDO MATRIZ DE CARACTERÍSTICAS MUNICIPAIS")
    print("-" * 40)
    
    # Agregar dados por município
    metricas_municipais = []
    
    for municipio in df_recente['id_municipio'].unique():
        # Dados do município
        df_mun = df_geral[df_geral['id_municipio'] == municipio]
        uf = df_mun['uf'].iloc[0]
        
        # Calcular métricas
        metricas = {
            'id_municipio': municipio,
            'uf': uf,
            'homicidio_media': df_mun['homicidio_doloso'].mean(),
            'homicidio_cv': df_mun['homicidio_doloso'].std() / (df_mun['homicidio_doloso'].mean() + 1e-6),
            'roubo_veiculo_media': df_mun['roubo_veiculo'].mean(),
            'furto_veiculo_media': df_mun['furto_veiculo'].mean(),
            'total_crimes_violentos': df_mun[['homicidio_doloso', 'roubo_total']].sum(axis=1).mean(),
            'proporcao_roubo_furto': df_mun['roubo_total'].sum() / (df_mun['furto_total'].sum() + 1),
            'tendencia_homicidios': calcular_tendencia(df_mun, 'homicidio_doloso'),
            'volatilidade_crimes': df_mun[['homicidio_doloso', 'roubo_total', 'furto_total']].std(axis=1).mean()
        }
        
        metricas_municipais.append(metricas)
    
    df_metricas = pd.DataFrame(metricas_municipais)
    
    # Carregar o mapeamento completo de IDs para nomes de municípios
    try:
        path_mapeamento = os.path.join(PASTA_DADOS, 'mapeamento_municipios.csv')
        
        df_temp = pd.read_csv(path_mapeamento)

        df_mapeamento = df_temp.iloc[:, 0].str.split(',', expand=True)
        

        df_mapeamento.columns = ['id_municipio', 'nome_municipio', 'regiao_uf', 'uf']

        df_mapeamento['id_municipio'] = pd.to_numeric(df_mapeamento['id_municipio'])

        municipios_rj_sp = pd.Series(
            df_mapeamento.nome_municipio.values,
            index=df_mapeamento.id_municipio
        ).to_dict()

    except FileNotFoundError:
        print("\nAVISO: Arquivo 'mapeamento_municipios.csv' não encontrado. Nomes não serão mapeados.")
        municipios_rj_sp = {}

    df_metricas['nome_municipio'] = df_metricas['id_municipio'].map(municipios_rj_sp).fillna('Outros')
    df_metricas['is_capital'] = df_metricas['id_municipio'].isin([3550308, 3304557]).astype(int)
    
    print(f"Matriz criada com {len(df_metricas)} municípios e {len(df_metricas.columns)-3} características")
    
    return df_metricas

def calcular_tendencia(df: pd.DataFrame, coluna: str) -> float:
    """
    Calcula tendência linear simples para uma série temporal.
    """
    if len(df) < 3:
        return 0
    
    x = np.arange(len(df))
    y = df[coluna].values
    
    if np.std(y) == 0:
        return 0
    
    coef = np.polyfit(x, y, 1)[0]
    return coef / (np.mean(y) + 1e-6)

def analise_pca_municipios(df_metricas: pd.DataFrame, pasta_saida: str):
    """
    Realiza análise de componentes principais para entender dimensões da criminalidade.
    """
    print("\n2. ANÁLISE DE COMPONENTES PRINCIPAIS (PCA)")
    print("-" * 40)
    
    # Preparar dados para PCA
    colunas_numericas = ['homicidio_media', 'homicidio_cv', 'roubo_veiculo_media', 
                        'furto_veiculo_media', 'total_crimes_violentos', 
                        'proporcao_roubo_furto', 'tendencia_homicidios', 'volatilidade_crimes']
    
    X = df_metricas[colunas_numericas].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Variância explicada
    var_explicada = pca.explained_variance_ratio_
    var_acumulada = np.cumsum(var_explicada)
    
    print("\nVariância explicada por componente:")
    for i in range(min(4, len(var_explicada))):
        print(f"  PC{i+1}: {var_explicada[i]:.2%} (acumulada: {var_acumulada[i]:.2%})")
    
    # Visualizar PCA
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scree plot
    ax1 = axes[0, 0]
    ax1.bar(range(1, len(var_explicada)+1), var_explicada, alpha=0.7)
    ax1.plot(range(1, len(var_explicada)+1), var_acumulada, 'ro-', linewidth=2)
    ax1.set_xlabel('Componente Principal')
    ax1.set_ylabel('Variância Explicada')
    ax1.set_title('Scree Plot - Variância Explicada')
    ax1.grid(True, alpha=0.3)
    
    # 2. Biplot PC1 vs PC2
    ax2 = axes[0, 1]
    
    # Pontos dos municípios
    scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], 
                         c=df_metricas['uf'].map({'SP': 'blue', 'RJ': 'red'}),
                         alpha=0.6, s=50)
    
    # Destacar principais municípios
    principais = df_metricas[df_metricas['nome_municipio'] != 'Outros']
    for idx, row in principais.iterrows():
        ax2.annotate(row['nome_municipio'], 
                    (X_pca[idx, 0], X_pca[idx, 1]),
                    fontsize=8, alpha=0.8)
    
    ax2.set_xlabel(f'PC1 ({var_explicada[0]:.1%})')
    ax2.set_ylabel(f'PC2 ({var_explicada[1]:.1%})')
    ax2.set_title('Biplot - PC1 vs PC2')
    ax2.grid(True, alpha=0.3)
    
    # 3. Loadings (contribuição das variáveis)
    ax3 = axes[1, 0]
    loadings = pca.components_[:2].T
    
    for i, (var, load) in enumerate(zip(colunas_numericas, loadings)):
        ax3.arrow(0, 0, load[0]*3, load[1]*3, 
                 head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax3.text(load[0]*3.5, load[1]*3.5, var, fontsize=9, ha='center')
    
    ax3.set_xlim(-4, 4)
    ax3.set_ylim(-4, 4)
    ax3.set_xlabel('PC1 Loadings')
    ax3.set_ylabel('PC2 Loadings')
    ax3.set_title('Contribuição das Variáveis')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.axvline(0, color='black', linewidth=0.5)
    
    # 4. Interpretação dos componentes
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Identificar principais contribuidores para cada PC
    pc1_vars = [(colunas_numericas[i], loadings[i, 0]) 
                for i in np.argsort(np.abs(loadings[:, 0]))[-3:]]
    pc2_vars = [(colunas_numericas[i], loadings[i, 1]) 
                for i in np.argsort(np.abs(loadings[:, 1]))[-3:]]
    
    interpretacao = f"""
    INTERPRETAÇÃO DOS COMPONENTES:
    
    PC1 ({var_explicada[0]:.1%} da variância):
    Principais contribuidores:
    {chr(8226)} {pc1_vars[2][0]}: {pc1_vars[2][1]:.2f}
    {chr(8226)} {pc1_vars[1][0]}: {pc1_vars[1][1]:.2f}
    {chr(8226)} {pc1_vars[0][0]}: {pc1_vars[0][1]:.2f}
    
    PC2 ({var_explicada[1]:.1%} da variância):
    Principais contribuidores:
    {chr(8226)} {pc2_vars[2][0]}: {pc2_vars[2][1]:.2f}
    {chr(8226)} {pc2_vars[1][0]}: {pc2_vars[1][1]:.2f}
    {chr(8226)} {pc2_vars[0][0]}: {pc2_vars[0][1]:.2f}
    """
    
    ax4.text(0.1, 0.9, interpretacao, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_saida, '07_analise_pca_municipios.png'), dpi=300)
    plt.close()
    
    return X_pca, pca

def clustering_hierarchico_municipios(df_metricas: pd.DataFrame, X_scaled: np.ndarray, pasta_saida: str):
    """
    Realiza clustering hierárquico para identificar grupos de municípios similares.
    """
    print("\n3. CLUSTERING HIERÁRQUICO DE MUNICÍPIOS")
    print("-" * 40)
    
    # Clustering hierárquico
    Z = linkage(X_scaled, method='ward')
    
    # Determinar número ótimo de clusters
    max_d = np.sort(Z[:, 2])[-10]  # Altura de corte para ~5-6 clusters
    clusters = fcluster(Z, max_d, criterion='distance')
    n_clusters = len(np.unique(clusters))
    
    df_metricas['cluster'] = clusters
    
    print(f"\nNúmero de clusters identificados: {n_clusters}")
    
    # Análise dos clusters
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Dendrograma
    ax1 = axes[0, 0]
    dendrogram(Z, ax=ax1, truncate_mode='level', p=5, 
              show_leaf_counts=True, no_labels=True)
    ax1.axhline(y=max_d, c='red', linestyle='--', label=f'{n_clusters} clusters')
    ax1.set_title('Dendrograma - Clustering Hierárquico')
    ax1.set_xlabel('Municípios')
    ax1.set_ylabel('Distância')
    ax1.legend()
    
    # 2. Distribuição dos clusters por estado
    ax2 = axes[0, 1]
    cluster_estado = pd.crosstab(df_metricas['cluster'], df_metricas['uf'])
    cluster_estado.plot(kind='bar', ax=ax2)
    ax2.set_title('Distribuição dos Clusters por Estado')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Número de Municípios')
    ax2.legend(title='Estado')
    
    # 3. Características médias por cluster
    ax3 = axes[1, 0]
    
    # Calcular médias normalizadas por cluster
    cluster_profiles = df_metricas.groupby('cluster')[
        ['homicidio_media', 'roubo_veiculo_media', 'furto_veiculo_media', 'volatilidade_crimes']
    ].mean()
    
    # Normalizar para melhor visualização
    cluster_profiles_norm = (cluster_profiles - cluster_profiles.mean()) / cluster_profiles.std()
    
    # Heatmap
    sns.heatmap(cluster_profiles_norm.T, annot=True, fmt='.2f', 
                cmap='RdBu_r', center=0, ax=ax3,
                xticklabels=[f'C{i}' for i in cluster_profiles.index])
    ax3.set_title('Perfil Normalizado dos Clusters')
    ax3.set_ylabel('Características')
    
    # 4. Descrição dos clusters
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    descricoes = []
    for cluster_id in sorted(df_metricas['cluster'].unique()):
        cluster_data = df_metricas[df_metricas['cluster'] == cluster_id]
        n_mun = len(cluster_data)
        principais = cluster_data[cluster_data['nome_municipio'] != 'Outros']['nome_municipio'].tolist()
        
        desc = f"Cluster {cluster_id} ({n_mun} municípios):"
        if principais:
            desc += f"\n  Inclui: {', '.join(principais[:3])}"
        
        # Característica dominante
        perfil = cluster_profiles.loc[cluster_id]
        dominante = perfil.idxmax()
        desc += f"\n  Característica: Alto {dominante.replace('_', ' ')}\n"
        
        descricoes.append(desc)
    
    ax4.text(0.1, 0.9, '\n'.join(descricoes), transform=ax4.transAxes,
            fontsize=10, verticalalignment='top')
    ax4.set_title('Descrição dos Clusters', fontsize=12, weight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_saida, '08_clustering_municipios.png'), dpi=300)
    plt.close()
    
    return df_metricas

def analise_hotspots(df_metricas: pd.DataFrame, pasta_saida: str):
    """
    Identifica e visualiza hotspots de criminalidade.
    """
    print("\n4. IDENTIFICAÇÃO DE HOTSPOTS")
    print("-" * 40)
    
    # Calcular índice composto de criminalidade
    df_metricas['indice_criminalidade'] = (
        0.4 * (df_metricas['homicidio_media'] / df_metricas['homicidio_media'].max()) +
        0.3 * (df_metricas['total_crimes_violentos'] / df_metricas['total_crimes_violentos'].max()) +
        0.3 * (df_metricas['volatilidade_crimes'] / df_metricas['volatilidade_crimes'].max())
    )
    
    # Identificar hotspots (top 10%)
    threshold = df_metricas['indice_criminalidade'].quantile(0.9)
    df_metricas['is_hotspot'] = df_metricas['indice_criminalidade'] > threshold
    
    # Visualização
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Distribuição do índice de criminalidade
    ax1 = axes[0]
    
    for uf in ['SP', 'RJ']:
        data = df_metricas[df_metricas['uf'] == uf]['indice_criminalidade']
        ax1.hist(data, bins=20, alpha=0.6, label=uf, density=True)
    
    ax1.axvline(threshold, color='red', linestyle='--', label=f'Hotspot threshold (90%)')
    ax1.set_xlabel('Índice de Criminalidade')
    ax1.set_ylabel('Densidade')
    ax1.set_title('Distribuição do Índice de Criminalidade')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Top hotspots
    ax2 = axes[1]
    
    top_hotspots = df_metricas.nlargest(15, 'indice_criminalidade')
    y_pos = np.arange(len(top_hotspots))
    
    colors = ['red' if uf == 'RJ' else 'blue' for uf in top_hotspots['uf']]
    bars = ax2.barh(y_pos, top_hotspots['indice_criminalidade'], color=colors, alpha=0.7)
    
    # Labels
    labels = []
    for idx, row in top_hotspots.iterrows():
        label = row['nome_municipio'] if row['nome_municipio'] != 'Outros' else f"ID: {row['id_municipio']}"
        labels.append(f"{label} ({row['uf']})")
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel('Índice de Criminalidade')
    ax2.set_title('Top 15 Hotspots de Criminalidade')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_saida, '09_hotspots_criminalidade.png'), dpi=300)
    plt.close()
    
    # Estatísticas dos hotspots
    print(f"\nTotal de hotspots identificados: {df_metricas['is_hotspot'].sum()}")
    print(f"  - SP: {df_metricas[(df_metricas['uf'] == 'SP') & df_metricas['is_hotspot']].shape[0]}")
    print(f"  - RJ: {df_metricas[(df_metricas['uf'] == 'RJ') & df_metricas['is_hotspot']].shape[0]}")
    
    return df_metricas

def gerar_relatorio_espacial(df_metricas: pd.DataFrame, pasta_saida: str):
    """
    Gera relatórios CSV para análise espacial no Power BI.
    """
    print("\n5. GERANDO RELATÓRIOS PARA POWER BI")
    print("-" * 40)
    
    # 1. Relatório de municípios com todas as métricas
    df_municipios = df_metricas.copy()
    
    # Define o número de casas decimais para cada coluna de métrica
    mapa_arredondamento = {
        'homicidio_media': 2,
        'homicidio_cv': 4,
        'roubo_veiculo_media': 2,
        'furto_veiculo_media': 2,
        'total_crimes_violentos': 2,
        'proporcao_roubo_furto': 4,
        'tendencia_homicidios': 4,
        'volatilidade_crimes': 2,
        'indice_criminalidade': 4
    }
    
    # Aplica o arredondamento usando o mapa definido
    df_municipios = df_municipios.round(decimals=mapa_arredondamento)
    
    df_municipios.to_csv(os.path.join(pasta_saida, 'analise_espacial_municipios.csv'), index=False)
    
    # 2. Relatório de clusters
    cluster_summary = df_metricas.groupby(['cluster', 'uf']).agg({
        'id_municipio': 'count',
        'homicidio_media': 'mean',
        'roubo_veiculo_media': 'mean',
        'furto_veiculo_media': 'mean',
        'indice_criminalidade': 'mean',
        'is_hotspot': 'sum'
    }).round(2)
    cluster_summary.columns = ['n_municipios', 'homicidio_media', 'roubo_veiculo_media', 
                              'furto_veiculo_media', 'indice_criminalidade_media', 'n_hotspots']
    cluster_summary.to_csv(os.path.join(pasta_saida, 'clusters_municipais_summary.csv'))
    
    # 3. Relatório de hotspots
    hotspots = df_metricas[df_metricas['is_hotspot']].copy()
    hotspots = hotspots.sort_values('indice_criminalidade', ascending=False)
    hotspots.to_csv(os.path.join(pasta_saida, 'hotspots_municipais.csv'), index=False)
    
    print("\nArquivos gerados:")
    print("  - analise_espacial_municipios.csv")
    print("  - clusters_municipais_summary.csv")
    print("  - hotspots_municipais.csv")
    
    return df_metricas

def criar_dashboard_final(df_metricas: pd.DataFrame, pasta_saida: str):
    """
    Cria um dashboard consolidado com os principais insights espaciais.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Dashboard de Análise Espacial - Municípios SP e RJ', fontsize=16, weight='bold')
    
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # 1. Comparação de médias entre capitais e interior
    ax1 = fig.add_subplot(gs[0, :2])
    
    df_capitais = df_metricas[df_metricas['is_capital'] == 1]
    df_interior = df_metricas[df_metricas['is_capital'] == 0]
    
    metricas_comp = ['homicidio_media', 'roubo_veiculo_media', 'furto_veiculo_media']
    x = np.arange(len(metricas_comp))
    width = 0.35
    
    medias_capitais = [df_capitais[m].mean() for m in metricas_comp]
    medias_interior = [df_interior[m].mean() for m in metricas_comp]
    
    ax1.bar(x - width/2, medias_capitais, width, label='Capitais', alpha=0.8)
    ax1.bar(x + width/2, medias_interior, width, label='Interior', alpha=0.8)
    
    ax1.set_xlabel('Tipo de Crime')
    ax1.set_ylabel('Média por Município')
    ax1.set_title('Comparação: Capitais vs Interior')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Homicídios', 'Roubo Veículos', 'Furto Veículos'])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Distribuição de clusters
    ax2 = fig.add_subplot(gs[0, 2])
    
    cluster_sizes = df_metricas['cluster'].value_counts().sort_index()
    ax2.pie(cluster_sizes, labels=[f'Cluster {i}' for i in cluster_sizes.index],
            autopct='%1.1f%%', startangle=90)
    ax2.set_title('Distribuição de Municípios por Cluster')
    
    # 3. Correlação entre tipos de crimes
    ax3 = fig.add_subplot(gs[1, :])
    
    corr_matrix = df_metricas[['homicidio_media', 'roubo_veiculo_media', 
                               'furto_veiculo_media', 'volatilidade_crimes',
                               'proporcao_roubo_furto']].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax3, square=True)
    ax3.set_title('Correlação entre Indicadores de Criminalidade')
    
    # 4. Insights principais
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Calcular insights
    n_hotspots_sp = len(df_metricas[(df_metricas['uf'] == 'SP') & df_metricas['is_hotspot']])
    n_hotspots_rj = len(df_metricas[(df_metricas['uf'] == 'RJ') & df_metricas['is_hotspot']])
    
    cluster_mais_violento = df_metricas.groupby('cluster')['indice_criminalidade'].mean().idxmax()
    cluster_mais_seguro = df_metricas.groupby('cluster')['indice_criminalidade'].mean().idxmin()
    
    insights_text = f"""
    PRINCIPAIS INSIGHTS DA ANÁLISE ESPACIAL:
    
    1. HOTSPOTS: Identificados {n_hotspots_sp + n_hotspots_rj} municípios críticos
       • SP: {n_hotspots_sp} hotspots ({n_hotspots_sp/len(df_metricas[df_metricas['uf']=='SP'])*100:.1f}% dos municípios)
       • RJ: {n_hotspots_rj} hotspots ({n_hotspots_rj/len(df_metricas[df_metricas['uf']=='RJ'])*100:.1f}% dos municípios)
    
    2. PADRÕES ESPACIAIS:
       • Capitais têm {(df_capitais['homicidio_media'].mean()/df_interior['homicidio_media'].mean()-1)*100:.1f}% mais homicídios que o interior
       • Cluster {cluster_mais_violento} é o mais violento (inclui grandes centros urbanos)
       • Cluster {cluster_mais_seguro} é o mais seguro (municípios menores do interior)
    
    3. CORRELAÇÕES IMPORTANTES:
       • Homicídios e roubos de veículos: r = {corr_matrix.loc['homicidio_media', 'roubo_veiculo_media']:.2f}
       • Volatilidade indica instabilidade criminal em {(df_metricas['volatilidade_crimes'] > df_metricas['volatilidade_crimes'].quantile(0.75)).sum()} municípios
    
    4. IMPLICAÇÕES PARA POLÍTICAS PÚBLICAS:
       • Necessidade de abordagens diferenciadas para capitais vs interior
       • Foco em municípios do Cluster {cluster_mais_violento} para redução de violência
       • Estratégias integradas para crimes correlacionados (homicídios + roubo de veículos)
    """
    
    ax4.text(0.05, 0.95, insights_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_saida, '10_dashboard_espacial_final.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    PASTA_DADOS = 'dados_tratados'
    PASTA_VISUALIZACOES = 'visualizacoes'
    
    try:
        # Carregar dados
        df_recente, df_geral = carregar_dados_municipais(PASTA_DADOS)
        
        # Criar matriz de características
        df_metricas = criar_matriz_crimes_municipios(df_recente, df_geral)
        
        # Preparar dados para análises
        colunas_analise = ['homicidio_media', 'homicidio_cv', 'roubo_veiculo_media', 
                          'furto_veiculo_media', 'total_crimes_violentos', 
                          'proporcao_roubo_furto', 'tendencia_homicidios', 'volatilidade_crimes']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_metricas[colunas_analise])
        
        # Executar análises
        X_pca, pca = analise_pca_municipios(df_metricas, PASTA_VISUALIZACOES)
        df_metricas = clustering_hierarchico_municipios(df_metricas, X_scaled, PASTA_VISUALIZACOES)
        df_metricas = analise_hotspots(df_metricas, PASTA_VISUALIZACOES)
        
        # Gerar relatórios
        gerar_relatorio_espacial(df_metricas, PASTA_DADOS)
        criar_dashboard_final(df_metricas, PASTA_VISUALIZACOES)
        
        print("\n" + "=" * 60)
        print("ANÁLISE ESPACIAL CONCLUÍDA COM SUCESSO!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERRO durante a execução: {e}")
        import traceback
        traceback.print_exc()
