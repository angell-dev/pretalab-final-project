#!/usr/bin/env python3
"""
Script 01 (v1.3): Processamento e Limpeza dos Dados do Disque 100
Versão corrigida para lidar com problemas de encoding nos nomes das colunas
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import re

warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# ESQUEMA MESTRE - Estrutura final desejada
# ==============================================================================
ESQUEMA_MESTRE = [
    'data_fato',
    'ano_fato', 
    'uf_ocorrencia',
    'municipio_ocorrencia',
    'grupo_vulneravel',
    'tipo_violacao',
    'vitima_sexo',
    'vitima_faixa_etaria',
    'vitima_raca_cor',
    'vitima_orientacao_sexual'
]

# ==============================================================================
# MAPEAMENTO EXPANDIDO - Lida com todas as variações de encoding
# ==============================================================================
MAPEAMENTO_MESTRE = {
    # Data - variações
    'data_da_denuncia': 'data_fato',
    'data_da_denúncia': 'data_fato',
    'data_de_cadastro': 'data_fato',
    'inicio_das_violacoes': 'data_fato',
    'início_das_violações': 'data_fato',
    'inicio_das_violaçoes': 'data_fato',
    'inicio_das_violaã§ãµes': 'data_fato',
    
    # UF - variações
    'uf': 'uf_ocorrencia',
    'estado': 'uf_ocorrencia',
    'uf_da_vitima': 'uf_ocorrencia',
    'uf_da_vítima': 'uf_ocorrencia',
    'uf_da_vã­tima': 'uf_ocorrencia',
    
    # Município - variações
    'municipio': 'municipio_ocorrencia',
    'município': 'municipio_ocorrencia',
    'municipio_da_vitima': 'municipio_ocorrencia',
    'município_da_vítima': 'municipio_ocorrencia',
    'municipio_da_vã­tima': 'municipio_ocorrencia',
    'municã­pio': 'municipio_ocorrencia',
    'municã­pio_da_vã­tima': 'municipio_ocorrencia',
    
    # Grupo vulnerável - variações
    'grupo_vulneravel': 'grupo_vulneravel',
    'grupo_vulnerável': 'grupo_vulneravel',
    'grupos_vulneraveis': 'grupo_vulneravel',
    'grupo_vulnerã¡vel': 'grupo_vulneravel',
    '(nenhum_nome_de_coluna)': 'grupo_vulneravel',
    
    # Violação - variações
    'violacao': 'tipo_violacao',
    'violação': 'tipo_violacao',
    'violacoes': 'tipo_violacao',
    'violações': 'tipo_violacao',
    'violacao_nome': 'tipo_violacao',
    'tipo_violacao': 'tipo_violacao',
    
    # Sexo/Gênero - variações
    'genero_da_vitima': 'vitima_sexo',
    'gênero_da_vítima': 'vitima_sexo',
    'sexo_da_vitima': 'vitima_sexo',
    'sexo_da_vítima': 'vitima_sexo',
    'gãªnero_da_vã­tima': 'vitima_sexo',
    'sexo_da_vã­tima': 'vitima_sexo',
    'genero_da_vã­tima': 'vitima_sexo',
    
    # Faixa etária - variações
    'faixa_etaria_da_vitima': 'vitima_faixa_etaria',
    'faixa_etária_da_vítima': 'vitima_faixa_etaria',
    'idade_vitima': 'vitima_faixa_etaria',
    'faixa_etã¡ria_da_vã­tima': 'vitima_faixa_etaria',
    
    # Raça/Cor - variações
    'raca_cor_da_vitima': 'vitima_raca_cor',
    'raça/cor_da_vítima': 'vitima_raca_cor',
    'raca\\cor_da_vitima': 'vitima_raca_cor',
    'raça\\cor_da_vítima': 'vitima_raca_cor',
    'raã§a_cor_da_vã­tima': 'vitima_raca_cor',
    'raã§a\\cor_da_vã­tima': 'vitima_raca_cor',
    'raã§a_cor_da_vã­tima': 'vitima_raca_cor',
    
    # Orientação sexual - variações
    'orientacao_sexual': 'vitima_orientacao_sexual',
    'orientação_sexual': 'vitima_orientacao_sexual',
    'orientacao_sexual_da_vitima': 'vitima_orientacao_sexual',
    'orientação_sexual_da_vítima': 'vitima_orientacao_sexual',
    'orientaã§ã£o_sexual_da_vã­tima': 'vitima_orientacao_sexual',
}

def limpar_nome_coluna(nome):
    """
    Limpa e normaliza o nome de uma coluna, removendo caracteres especiais
    e padronizando o formato.
    """
    if pd.isna(nome):
        return 'coluna_sem_nome'
    
    # Converte para string e lowercase
    nome = str(nome).lower().strip()
    
    # Remove o prefixo BOM se existir
    nome = nome.replace('ï»¿', '').replace('\ufeff', '')
    
    # Substitui caracteres especiais comuns por suas versões normais
    substituicoes = {
        'ã¡': 'a', 'ã¢': 'a', 'ã£': 'a', 'ã¤': 'a', 'ã ': 'a',
        'ã©': 'e', 'ãª': 'e', 'ã¨': 'e', 'ã«': 'e',
        'ã­': 'i', 'ã¬': 'i', 'ã®': 'i', 'ã¯': 'i',
        'ã³': 'o', 'ã²': 'o', 'ã´': 'o', 'ãµ': 'o', 'ã¶': 'o',
        'ãº': 'u', 'ã¹': 'u', 'ã»': 'u', 'ã¼': 'u',
        'ã§': 'c', 'ã±': 'n',
        'á': 'a', 'à': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a',
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
        'ó': 'o', 'ò': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o',
        'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u',
        'ç': 'c', 'ñ': 'n'
    }
    
    for char_antigo, char_novo in substituicoes.items():
        nome = nome.replace(char_antigo, char_novo)
    
    # Substitui espaços e caracteres especiais por underscore
    nome = re.sub(r'[^a-z0-9]+', '_', nome)
    
    # Remove underscores no início e fim
    nome = nome.strip('_')
    
    # Remove underscores duplicados
    nome = re.sub(r'_+', '_', nome)
    
    return nome

def mapear_coluna(nome_limpo):
    """
    Mapeia o nome limpo da coluna para o esquema mestre.
    """
    # Primeiro tenta mapeamento direto
    if nome_limpo in MAPEAMENTO_MESTRE:
        return MAPEAMENTO_MESTRE[nome_limpo]
    
    # Tenta encontrar correspondências parciais
    for chave, valor in MAPEAMENTO_MESTRE.items():
        if chave in nome_limpo or nome_limpo in chave:
            return valor
    
    # Mapeamentos especiais baseados em palavras-chave
    if 'grupo' in nome_limpo and 'vulner' in nome_limpo:
        return 'grupo_vulneravel'
    if 'data' in nome_limpo and ('cadastro' in nome_limpo or 'denuncia' in nome_limpo):
        return 'data_fato'
    if 'uf' in nome_limpo and 'vitima' not in nome_limpo:
        return 'uf_ocorrencia'
    if 'municipio' in nome_limpo and 'vitima' not in nome_limpo:
        return 'municipio_ocorrencia'
    if 'violacao' in nome_limpo or 'violacoes' in nome_limpo:
        return 'tipo_violacao'
    if ('genero' in nome_limpo or 'sexo' in nome_limpo) and 'vitima' in nome_limpo:
        return 'vitima_sexo'
    if 'orientacao' in nome_limpo and 'sexual' in nome_limpo:
        return 'vitima_orientacao_sexual'
    if ('faixa' in nome_limpo and 'etaria' in nome_limpo) or 'idade_vitima' in nome_limpo:
        return 'vitima_faixa_etaria'
    if 'raca' in nome_limpo or 'cor' in nome_limpo:
        return 'vitima_raca_cor'
    
    return None

def processar_arquivo(arquivo, dtype_map):
    """
    Processa um arquivo individual do Disque 100.
    """
    print(f"   🔎 Processando: {arquivo.name}")
    
    try:
        # Lê o arquivo
        df = pd.read_csv(arquivo, sep=';', encoding='latin-1', 
                        on_bad_lines='skip', dtype=dtype_map)
        
        print(f"      ✓ Lido: {len(df):,} registros")
        
        # Cria um novo dataframe com as colunas mapeadas
        df_mapeado = pd.DataFrame()
        
        # Mapeia cada coluna
        colunas_mapeadas = 0
        for col_original in df.columns:
            col_limpa = limpar_nome_coluna(col_original)
            col_mapeada = mapear_coluna(col_limpa)
            
            if col_mapeada and col_mapeada in ESQUEMA_MESTRE:
                df_mapeado[col_mapeada] = df[col_original]
                colunas_mapeadas += 1
        
        print(f"      ✓ Mapeadas {colunas_mapeadas} colunas para o esquema mestre")
        
        # Adiciona colunas faltantes com NaN
        for col in ESQUEMA_MESTRE:
            if col not in df_mapeado.columns:
                df_mapeado[col] = np.nan
        
        # Reordena as colunas conforme o esquema
        df_mapeado = df_mapeado[ESQUEMA_MESTRE]
        
        return df_mapeado
        
    except Exception as e:
        print(f"      ⚠️ Erro ao processar {arquivo.name}: {e}")
        return None

def processar_disque100(base_path):
    """
    Processa todos os arquivos do Disque 100.
    """
    print("🚀 Iniciando o processamento do Disque 100 (v1.3 - Correção Encoding)...")
    
    disque100_path = base_path / "dados_brutos" / "disque100"
    arquivos_csv = list(disque100_path.glob("*.csv"))
    
    if not arquivos_csv:
        print("❌ ERRO: Nenhum arquivo CSV do Disque 100 encontrado.")
        return
    
    # Define tipos de dados para colunas problemáticas
    colunas_com_tipos_mistos = [20, 21, 22, 25, 34, 35, 41, 42, 43, 46, 52, 55, 56, 57, 59, 60]
    dtype_map = {col: str for col in colunas_com_tipos_mistos}
    
    # Processa cada arquivo
    lista_dfs = []
    for arquivo in sorted(arquivos_csv):
        df_processado = processar_arquivo(arquivo, dtype_map)
        if df_processado is not None:
            lista_dfs.append(df_processado)
    
    if not lista_dfs:
        print("❌ ERRO: Nenhum arquivo pôde ser processado.")
        return
    
    # Unifica todos os dataframes
    df_completo = pd.concat(lista_dfs, ignore_index=True)
    print(f"\n✅ Arquivos unificados! Total de {len(df_completo):,} registros.")
    
    # Debug: Mostra uma amostra dos dados para verificar o conteúdo
    print("\n📊 Amostra dos dados unificados:")
    print(f"   Colunas: {list(df_completo.columns)}")
    print(f"   Valores únicos em 'uf_ocorrencia': {df_completo['uf_ocorrencia'].value_counts().head(10).to_dict()}")
    print(f"   Valores únicos em 'grupo_vulneravel': {df_completo['grupo_vulneravel'].value_counts().head(10).to_dict()}")
    
    # Filtra para o Sudeste
    estados_sudeste = ['SP', 'SÃO PAULO', 'São Paulo', 'MG', 'MINAS GERAIS', 'Minas Gerais', 
                       'RJ', 'RIO DE JANEIRO', 'Rio de Janeiro', 'ES', 'ESPÍRITO SANTO', 'Espírito Santo',
                       'SAO PAULO', 'ESPIRITO SANTO']
    
    # Normaliza os valores de UF antes de filtrar
    df_completo['uf_ocorrencia_norm'] = df_completo['uf_ocorrencia'].str.upper().str.strip()
    
    df_sudeste = df_completo[df_completo['uf_ocorrencia_norm'].isin(estados_sudeste)].copy()
    print(f"\n✅ Filtrado para o Sudeste: {len(df_sudeste):,} registros.")
    
    if len(df_sudeste) == 0:
        print("   ⚠️ Debug: Verificando valores de UF...")
        valores_uf = df_completo['uf_ocorrencia'].value_counts().head(20)
        print(f"   Primeiros 20 valores: {valores_uf.to_dict()}")
    
    # Filtra para LGBTQIA+
    if len(df_sudeste) > 0:
        # Cria uma coluna auxiliar para busca case-insensitive
        df_sudeste['grupo_vulneravel_norm'] = df_sudeste['grupo_vulneravel'].astype(str).str.upper()
        
        # Busca por diferentes variações de LGBTQIA+
        mask_lgbt = (
            df_sudeste['grupo_vulneravel_norm'].str.contains('LGBT', na=False) |
            df_sudeste['grupo_vulneravel_norm'].str.contains('GAY', na=False) |
            df_sudeste['grupo_vulneravel_norm'].str.contains('LÉSBICA', na=False) |
            df_sudeste['grupo_vulneravel_norm'].str.contains('TRAVESTI', na=False) |
            df_sudeste['grupo_vulneravel_norm'].str.contains('TRANSEXUAL', na=False) |
            df_sudeste['grupo_vulneravel_norm'].str.contains('BISSEXUAL', na=False)
        )
        
        df_lgbtqia = df_sudeste[mask_lgbt].copy()
        print(f"✅ Filtrado para LGBTQIA+: {len(df_lgbtqia):,} registros encontrados.")
        
        if len(df_lgbtqia) == 0:
            print("   ⚠️ Debug: Verificando valores de grupo vulnerável...")
            valores_grupo = df_sudeste['grupo_vulneravel'].value_counts().head(20)
            print(f"   Primeiros 20 valores: {valores_grupo.to_dict()}")
    
    # Se encontrou registros, salva o resultado
    if len(df_lgbtqia) > 0:
        # Limpa os dados
        df_lgbtqia['data_fato'] = pd.to_datetime(df_lgbtqia['data_fato'], errors='coerce')
        df_lgbtqia['ano_fato'] = df_lgbtqia['data_fato'].dt.year
        
        # Remove as colunas auxiliares
        df_lgbtqia = df_lgbtqia.drop(columns=['uf_ocorrencia_norm', 'grupo_vulneravel_norm'], errors='ignore')
        
        # Salva o resultado
        output_path = base_path / "dados_tratados" / "violencia_lgbtqia_disque100_sudeste.csv"
        output_path.parent.mkdir(exist_ok=True)
        df_lgbtqia.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n💾 Dataset processado salvo em:\n   {output_path}")
    else:
        print("\n⚠️ Nenhum registro LGBTQIA+ encontrado. Verifique os filtros e os dados de entrada.")

def encontrar_caminho_projeto():
    """Encontra o caminho raiz do projeto."""
    script_path = Path(__file__).resolve().parent
    project_base_path = script_path
    for _ in range(4):
        if (project_base_path / "dados_brutos").exists():
            return project_base_path
        project_base_path = project_base_path.parent
    raise FileNotFoundError("Pasta 'dados_brutos' não encontrada.")

def main():
    try:
        base_path = encontrar_caminho_projeto()
        processar_disque100(base_path)
    except FileNotFoundError as e:
        print(f"❌ ERRO CRÍTICO: {e}")
    except Exception as e:
        print(f"❌ Ocorreu um erro inesperado: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()