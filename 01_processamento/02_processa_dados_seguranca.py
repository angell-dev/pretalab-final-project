#!/usr/bin/env python3
"""
Script 02: Processamento e Unificação dos Dados de Segurança Pública (SP & RJ)

Este script realiza as seguintes etapas:
1.  Lê os dados brutos de segurança pública do Rio de Janeiro (ISP).
2.  Lê os dados brutos de segurança pública de São Paulo (SSP).
3.  Para cada estado:
    a. Seleciona as colunas de interesse (homicídios, roubos, furtos).
    b. Renomeia as colunas para um esquema padronizado para permitir a comparação.
    c. Garante que os tipos de dados estejam corretos.
4.  Unifica os dados de SP e RJ em um único DataFrame.
5.  Salva o dataset consolidado na pasta 'dados_tratados'.
"""
import pandas as pd
from pathlib import Path

def encontrar_caminho_projeto():
    """Encontra o caminho raiz do projeto subindo a partir do local do script."""
    script_path = Path(__file__).resolve().parent
    project_base_path = script_path
    for _ in range(4):
        if (project_base_path / "dados_brutos").exists():
            return project_base_path
        project_base_path = project_base_path.parent
    raise FileNotFoundError("Pasta 'dados_brutos' não encontrada. Verifique a estrutura do projeto.")

def processar_dados_rj(path_arquivo):
    """Processa e padroniza os dados de segurança do RJ."""
    print(f"   -> Processando dados do Rio de Janeiro...")
    
    df_rj = pd.read_csv(path_arquivo, encoding='latin-1', sep=',')

    # Mapeamento para o esquema padrão
    mapa_colunas_rj = {
        'ano': 'ano',
        'mes': 'mes',
        'id_municipio': 'id_municipio',
        'quantidade_homicidio_doloso': 'homicidio_doloso',
        'quantidade_roubo_veiculo': 'roubo_veiculo',
        'quantidade_furto_veiculos': 'furto_veiculo', # Nome da coluna em RJ
        'quantidade_total_roubos': 'roubo_total',
        'quantidade_total_furtos': 'furto_total'
    }
    
    df_rj_selecionado = df_rj[list(mapa_colunas_rj.keys())]
    df_rj_padronizado = df_rj_selecionado.rename(columns=mapa_colunas_rj)
    
    # Adiciona a coluna UF
    df_rj_padronizado['uf'] = 'RJ'
    
    print(f"      ✓ {len(df_rj_padronizado):,} registros do RJ padronizados.")
    return df_rj_padronizado

def processar_dados_sp(path_arquivo):
    """Processa e padroniza os dados de segurança de SP."""
    print(f"   -> Processando dados de São Paulo...")
    
    df_sp = pd.read_csv(path_arquivo, encoding='latin-1', sep=',')

    # Seleciona colunas de interesse para o esquema
    colunas_sp_interesse = [
        'ano', 'mes', 'id_municipio', 'homicidio_doloso',
        'roubo_de_veiculo', 'roubo_outros',
        'furto_de_veiculo', 'furto_outros'
    ]
    df_sp_selecionado = df_sp[colunas_sp_interesse].copy()

    # Cria totais para equiparar com os dados do RJ
    # Converte para numérico, tratando erros que possam surgir
    for col in ['roubo_outros', 'roubo_de_veiculo', 'furto_outros', 'furto_de_veiculo']:
        df_sp_selecionado[col] = pd.to_numeric(df_sp_selecionado[col], errors='coerce')
    
    df_sp_selecionado['roubo_total'] = df_sp_selecionado['roubo_outros'].fillna(0) + df_sp_selecionado['roubo_de_veiculo'].fillna(0)
    df_sp_selecionado['furto_total'] = df_sp_selecionado['furto_outros'].fillna(0) + df_sp_selecionado['furto_de_veiculo'].fillna(0)
    
    # Renomeia para o esquema padrão
    df_sp_padronizado = df_sp_selecionado.rename(columns={
        'roubo_de_veiculo': 'roubo_veiculo',
        'furto_de_veiculo': 'furto_veiculo'
    })

    # Adiciona a coluna UF
    df_sp_padronizado['uf'] = 'SP'

    # Seleciona apenas as colunas do esquema final para garantir consistência
    colunas_finais = [
        'ano', 'mes', 'id_municipio', 'uf', 'homicidio_doloso', 
        'roubo_veiculo', 'furto_veiculo', 'roubo_total', 'furto_total'
    ]
    
    print(f"      ✓ {len(df_sp_padronizado):,} registros de SP padronizados.")
    return df_sp_padronizado[colunas_finais]

def processar_seguranca_publica(base_path):
    """
    Orquestra a unificação dos dados de segurança de SP e RJ.
    """
    print("🚀 Iniciando o processamento dos dados de Segurança Pública (SP & RJ)...")
    
    # Caminhos para os arquivos brutos
    path_rj = base_path / "dados_brutos" / "seguranca_publica" / "br_rj_isp_estatisticas_seguranca_evolucao_mensal_municipio.csv"
    path_sp = base_path / "dados_brutos" / "seguranca_publica" / "br_sp_gov_ssp_ocorrencias_registradas (1).csv"

    if not path_rj.exists() or not path_sp.exists():
        print("❌ ERRO: Arquivos de segurança de SP ou RJ não encontrados na pasta 'dados_brutos/seguranca_publica/'.")
        return

    # Processa cada estado
    df_rj = processar_dados_rj(path_rj)
    df_sp = processar_dados_sp(path_sp)
    
    # Unifica os DataFrames
    df_unificado = pd.concat([df_rj, df_sp], ignore_index=True)
    print(f"\n✅ Dados de SP e RJ unificados. Total de {len(df_unificado):,} registros.")

    # Garante que as colunas de crimes sejam numéricas
    for col in ['homicidio_doloso', 'roubo_veiculo', 'furto_veiculo', 'roubo_total', 'furto_total']:
        df_unificado[col] = pd.to_numeric(df_unificado[col], errors='coerce').fillna(0).astype(int)

    # Salva o resultado
    output_path = base_path / "dados_tratados" / "seguranca_publica_sp_rj_consolidado.csv"
    output_path.parent.mkdir(exist_ok=True)
    
    df_unificado.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n💾 SUCESSO! Dataset de Segurança Pública consolidado foi salvo em:\n   {output_path}")

def main():
    try:
        base_path = encontrar_caminho_projeto()
        processar_seguranca_publica(base_path)
    except FileNotFoundError as e:
        print(f"❌ ERRO CRÍTICO: {e}")
    except Exception as e:
        print(f"❌ Ocorreu um erro inesperado: {e}")

if __name__ == '__main__':
    main()