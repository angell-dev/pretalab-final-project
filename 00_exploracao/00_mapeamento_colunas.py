#!/usr/bin/env python3
"""
Script 2: Mapeador de Colunas
Gera um dicionário de dados de todas as fontes para planejar a unificação.
"""
import os
import pandas as pd
import gzip
from pathlib import Path
from datetime import datetime

# Reutilizando a lógica robusta de leitura do explorador
def ler_cabecalho_seguro(filepath):
    """Lê apenas o cabeçalho de um arquivo para obter as colunas de forma eficiente."""
    filepath = Path(filepath)
    try:
        if filepath.suffix == '.gz':
            with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
                return pd.read_csv(f, sep=',', nrows=0).columns.tolist()
        elif filepath.suffix == '.xlsx':
            return pd.read_excel(filepath, nrows=0).columns.tolist()
        elif filepath.suffix == '.csv':
            if 'disque100' in str(filepath).lower():
                return pd.read_csv(filepath, sep=';', encoding='latin-1', nrows=0).columns.tolist()
            
            encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1']
            for enc in encodings_to_try:
                try:
                    return pd.read_csv(filepath, sep=None, engine='python', encoding=enc, nrows=0).columns.tolist()
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
            raise Exception("Não foi possível ler o cabeçalho do CSV.")
        else:
            return [f"Formato não suportado: {filepath.suffix}"]
    except Exception as e:
        return [f"ERRO AO LER ARQUIVO: {e}"]

def mapear_colunas(base_path):
    """
    Varre os diretórios de dados e gera um arquivo de texto com o mapeamento das colunas.
    """
    base_path = Path(base_path)
    dados_brutos = base_path / "dados_brutos"
    dados_tratados = base_path / "dados_tratados"
    relatorios_path = base_path / "relatorios"
    relatorios_path.mkdir(exist_ok=True)
    
    mapa_colunas = []
    
    print("🔎 Mapeando colunas dos DADOS BRUTOS...")
    for root, _, files in os.walk(dados_brutos):
        for file in sorted(files):
            if not file.startswith('.') and Path(file).suffix not in ['.html']:
                filepath = Path(root) / file
                print(f"  -> Lendo cabeçalho de: {file}")
                colunas = ler_cabecalho_seguro(filepath)
                mapa_colunas.append({
                    "caminho_arquivo": str(filepath.relative_to(base_path)),
                    "colunas": colunas
                })

    print("\n🔎 Mapeando colunas dos DADOS TRATADOS...")
    if dados_tratados.exists():
        for root, _, files in os.walk(dados_tratados):
            for file in sorted(files):
                if not file.startswith('.') and Path(file).suffix not in ['.html']:
                    filepath = Path(root) / file
                    print(f"  -> Lendo cabeçalho de: {file}")
                    colunas = ler_cabecalho_seguro(filepath)
                    mapa_colunas.append({
                        "caminho_arquivo": str(filepath.relative_to(base_path)),
                        "colunas": colunas
                    })

    # Salvar o relatório de mapeamento
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = relatorios_path / f"mapeamento_colunas_{timestamp}.txt"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("### 🗺️ Dicionário de Dados - Mapeamento de Colunas ###\n")
        f.write(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        for item in mapa_colunas:
            f.write(f"📂 ARQUIVO: {item['caminho_arquivo']}\n")
            # Imprime colunas de forma legível
            col_str = ',\n'.join([f"    '{col}'" for col in item['colunas']])
            f.write(f"   [\n{col_str}\n   ]\n\n")
    
    print(f"\n✅ Dicionário de Dados salvo em: {output_path}")
    return output_path

def main():
    try:
        script_path = Path(__file__).resolve().parent
        project_base_path = script_path
        for _ in range(3):
            if (project_base_path / "dados_brutos").exists(): break
            project_base_path = project_base_path.parent
        else:
            raise FileNotFoundError("Não foi possível localizar a pasta 'dados_brutos'.")
        print(f"Diretório do projeto detectado em: {project_base_path}")
    except Exception:
        project_base_path = Path.cwd()
        print(f"Executando no diretório de trabalho atual: {project_base_path}")

    mapear_colunas(project_base_path)

if __name__ == '__main__':
    main()