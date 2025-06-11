#!/usr/bin/env python3
"""
Script 03 (v3.4): Enriquecimento e Finalização (Mapeamento Robusto Final)
Versão final que garante o mapeamento correto dos 4 estados do Sudeste.
"""
import pandas as pd
from pathlib import Path
import warnings
import unicodedata

warnings.simplefilter(action='ignore', category=FutureWarning)

def encontrar_caminho_projeto():
    """Encontra o caminho raiz do projeto de forma robusta."""
    script_path = Path(__file__).resolve().parent
    project_base_path = script_path
    for _ in range(4):
        if (project_base_path / "dados_brutos").exists(): return project_base_path
        project_base_path = project_base_path.parent
    raise FileNotFoundError("Pasta 'dados_brutos' não encontrada.")

def normalizar_texto(series):
    """Normaliza uma coluna de texto para comparação (caixa alta, sem acentos)."""
    return series.astype(str).str.upper().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

def enriquecer_datasets(base_path):
    """Orquestra o processo de enriquecimento e finalização dos datasets."""
    print("🚀 Iniciando a etapa final de enriquecimento (v3.4 - Mapeamento Final e Robusto)...")

    # --- 1. CARREGAR OS DATASETS ---
    path_violencia_lgbtqia = base_path / "dados_tratados" / "violencia_lgbtqia_disque100_sudeste.csv"
    path_seguranca = base_path / "dados_tratados" / "seguranca_publica_sp_rj_consolidado.csv"
    path_populacao = base_path / "dados_brutos" / "ibge" / "br_ibge_populacao_municipio.csv.gz"
    path_diretorio_mun = base_path / "dados_brutos" / "ibge" / "br_bd_diretorios_brasil_municipio.csv.gz"

    try:
        df_violencia = pd.read_csv(path_violencia_lgbtqia)
        df_seguranca = pd.read_csv(path_seguranca)
        df_populacao = pd.read_csv(path_populacao)
        df_diretorio = pd.read_csv(path_diretorio_mun, usecols=['id_municipio', 'nome', 'sigla_uf'])
        print("   ✅ Datasets carregados com sucesso.")
    except FileNotFoundError as e:
        print(f"❌ ERRO CRÍTICO: {e}")
        return

    # --- 2. PREPARAR DADOS DE POPULAÇÃO E MAPA DE MUNICÍPIOS ---
    ano_referencia_pop = df_populacao['ano'].max()
    df_pop_ref = df_populacao[df_populacao['ano'] == ano_referencia_pop][['id_municipio', 'populacao']].copy()
    print(f"   -> Usando população de referência do ano: {ano_referencia_pop}")

    df_diretorio['chave_unica'] = normalizar_texto(df_diretorio['nome']) + " - " + normalizar_texto(df_diretorio['sigla_uf'])
    mapa_chave_para_id = df_diretorio.drop_duplicates(subset=['chave_unica']).set_index('chave_unica')['id_municipio']
    print("   -> Mapa 'Nome do Município - UF -> ID' criado.")

    # --- 3. ENRIQUECER DADOS DE VIOLÊNCIA LGBTQIA+ ---
    print("   -> Processando dados de violência LGBTQIA+...")

    # Mapeia nomes completos para siglas de forma robusta
    map_uf_sigla = {'SAO PAULO': 'SP', 'MINAS GERAIS': 'MG', 'RIO DE JANEIRO': 'RJ', 'ESPIRITO SANTO': 'ES'}
    df_violencia['sigla_uf'] = normalizar_texto(df_violencia['uf_ocorrencia']).map(map_uf_sigla)
    
    # Agrupamos por UF e Município para garantir a unicidade
    df_violencia_agg = df_violencia.groupby(['municipio_ocorrencia', 'uf_ocorrencia', 'sigla_uf']).size().reset_index(name='total_denuncias')

    # Criar a chave composta para o merge
    df_violencia_agg['chave_unica'] = normalizar_texto(df_violencia_agg['municipio_ocorrencia']) + " - " + df_violencia_agg['sigla_uf']
    df_violencia_agg['id_municipio'] = df_violencia_agg['chave_unica'].map(mapa_chave_para_id)
    
    # Diagnóstico de Mapeamento
    total_municipios_unicos = len(df_violencia_agg)
    df_violencia_agg.dropna(subset=['id_municipio'], inplace=True)
    total_mapeados = len(df_violencia_agg)
    print(f"      ✓ Diagnóstico de Mapeamento: {total_mapeados} de {total_municipios_unicos} municípios foram identificados com sucesso ({total_mapeados/total_municipios_unicos:.1%}).")

    df_violencia_agg['id_municipio'] = df_violencia_agg['id_municipio'].astype(int)

    # Juntar com a população e calcular a taxa
    df_violencia_final = pd.merge(df_violencia_agg, df_pop_ref, on='id_municipio', how='left')
    df_violencia_final.dropna(subset=['populacao'], inplace=True) 
    
    df_violencia_final['taxa_denuncias_100k_hab'] = (df_violencia_final['total_denuncias'] / df_violencia_final['populacao']) * 100000
    
    # --- 4. ENRIQUECER DADOS DE SEGURANÇA PÚBLICA ---
    print("   -> Processando dados de Segurança Pública...")
    df_seguranca_final = pd.merge(df_seguranca, df_pop_ref, on='id_municipio', how='left')
    
    # --- 5. SALVAR ARQUIVOS FINAIS ---
    output_path_final = base_path / "dados_finais"
    output_path_final.mkdir(exist_ok=True)
    
    path_out_violencia = output_path_final / "analise_violencia_lgbtqia_municipal.csv"
    df_violencia_final.to_csv(path_out_violencia, index=False, encoding='utf-8', float_format='%.4f')
    print(f"\n💾 Dataset final de violência LGBTQIA+ (corrigido) salvo em: {path_out_violencia}")

    path_out_seguranca = output_path_final / "analise_seguranca_sp_rj_mensal.csv"
    df_seguranca_final.to_csv(path_out_seguranca, index=False, encoding='utf-8')
    print(f"💾 Dataset final de Segurança Pública salvo em: {path_out_seguranca}")


def main():
    try:
        base_path = encontrar_caminho_projeto()
        enriquecer_datasets(base_path)
        print("\n🎉 Processo de Engenharia de Dados CONCLUÍDO (v3.4)! 🎉")
        print("   Base de dados corrigida e pronta para a análise completa.")
    except Exception as e:
        print(f"❌ Ocorreu um erro inesperado: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()