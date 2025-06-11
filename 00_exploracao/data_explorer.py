#!/usr/bin/env python3
"""
Script 1 (v3.3): Explorador Universal de Dados Robusto
Análise completa de todos os arquivos para projeto de Segurança Pública e LGBTQIA+
com tratamento de erros avançado, análise de qualidade e salvamento organizado.
"""
import os
import pandas as pd
import numpy as np
import json
import gzip
from pathlib import Path
from datetime import datetime
import warnings
from openpyxl import load_workbook

# Ignora avisos comuns que não impedem a execução
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
warnings.filterwarnings('ignore', category=pd.errors.ParserWarning)

class DataExplorer:
    """
    Uma classe robusta para explorar e analisar um diretório de arquivos de dados
    de diversas fontes e formatos, gerando relatórios detalhados de qualidade e conteúdo.
    """
    def __init__(self, project_base_path):
        self.base_path = Path(project_base_path)
        self.dados_brutos = self.base_path / "dados_brutos"
        self.dados_tratados = self.base_path / "dados_tratados"
        self.relatorios_path = self.base_path / "relatorios"
        self.relatorios_path.mkdir(exist_ok=True)
        self.relatorio = {
            'metadados': {
                'data_analise': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'versao': '3.3 - Leitura Robusta Final',
                'analista': 'PretaLab Data Science'
            },
            'arquivos_analisados': [],
            'resumo_executivo': {
                'total_arquivos_processados': 0,
                'total_arquivos_com_erro': 0,
                'categorias': {}
            },
            'alertas_gerais': []
        }

    def _ler_arquivo_seguro(self, filepath):
        """Lê arquivos com tratamento robusto de erros de formato e encoding."""
        filepath = Path(filepath)
        try:
            if filepath.suffix == '.gz':
                with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
                    return pd.read_csv(f, sep=','), None
            elif filepath.suffix == '.xlsx':
                return pd.read_excel(filepath), None
            elif filepath.suffix == '.json':
                 return pd.read_json(filepath, orient='records'), None
            elif filepath.suffix == '.csv':
                # Caso especial: Arquivos CSV do Disque 100 usam ';' como separador
                if 'disque100' in str(filepath).lower():
                    return pd.read_csv(filepath, sep=';', encoding='latin-1', on_bad_lines='skip'), None
                
                # Para outros CSVs, tenta uma combinação de encodings
                encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1']
                for enc in encodings_to_try:
                    try:
                        return pd.read_csv(filepath, sep=None, engine='python', encoding=enc, on_bad_lines='skip'), None
                    except UnicodeDecodeError:
                        continue # Tenta o próximo encoding
                # Se todos falharem, retorna o erro do último
                raise UnicodeDecodeError(f"Não foi possível decodificar o arquivo com {encodings_to_try}")
            else:
                return None, f"Formato de arquivo não suportado: {filepath.suffix}"
        except Exception as e:
            return None, str(e)

    def _analisar_qualidade(self, df, filepath):
        if df is None or df.empty: return {}
        qualidade = {}
        try:
            qualidade['registros_duplicados'] = int(df.duplicated().sum())
            total_celulas = df.size
            qualidade['percentual_nulos_geral'] = round(df.isnull().sum().sum() * 100 / total_celulas, 2) if total_celulas > 0 else 0
            qualidade['memoria_usada_mb'] = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        except Exception as e:
            qualidade['erro'] = str(e)
        return qualidade

    def _analise_profunda_disque100(self, df):
        insights = {}
        opcoes_coluna = ['GRUPO_VULNERAVEL', 'GRUPOS_VULNERAVEIS', 'DS_GRUPO_VULNERAVEL']
        coluna_grupo_vulneravel = next((col for col in opcoes_coluna if col in df.columns), None)

        if not coluna_grupo_vulneravel:
            return {"erro": "Coluna de grupo vulnerável não encontrada."}
        
        df_lgbt = df[df[coluna_grupo_vulneravel].astype(str).str.contains('LGBT', na=False, case=False)]
        insights['total_registros_lgbtqia'] = len(df_lgbt)
        return insights

    def _categorizar_arquivo(self, filepath):
        path_str = str(filepath).lower()
        if 'ibge' in path_str and 'populacao' in path_str: return 'Demográfico (IBGE)'
        if any(keyword in path_str for keyword in ['lgbt', 'ggb', 'lgbtqiafobia']): return 'LGBTQIA+'
        if 'disque100' in path_str: return 'Disque 100'
        if any(keyword in path_str for keyword in ['ssp', 'isp', 'seguranca']): return 'Segurança Pública'
        if 'sahuv' in path_str: return 'Direitos Humanos (ES)'
        return 'Outros'

    def explorar_arquivo(self, filepath):
        print(f"🔎 Processando: {filepath.name}")
        df, erro = self._ler_arquivo_seguro(filepath)

        if erro or df is None:
            print(f"   ❌ Erro: {erro or 'Não foi possível ler o arquivo.'}")
            self.relatorio['resumo_executivo']['total_arquivos_com_erro'] += 1
            info_arquivo = {'nome_arquivo': str(filepath.relative_to(self.base_path)), 'status': 'ERRO', 'detalhe_erro': erro}
        else:
            print(f"   ✅ LIDO COM SUCESSO - {len(df):,} linhas.")
            info_arquivo = {
                'nome_arquivo': str(filepath.relative_to(self.base_path)), 'status': 'OK',
                'categoria': self._categorizar_arquivo(filepath),
                'dimensoes': {'linhas': len(df), 'colunas': len(df.columns)},
                'colunas': df.columns.tolist(),
                'preview_dados_llm': df.head(5).to_string(),
                'analise_qualidade': self._analisar_qualidade(df, filepath)
            }
            if 'disque100' in info_arquivo['nome_arquivo'].lower():
                info_arquivo['insights_especificos'] = self._analise_profunda_disque100(df)

        self.relatorio['arquivos_analisados'].append(info_arquivo)

    def explorar_diretorio(self, diretorio):
        for root, _, files in os.walk(diretorio):
            for file in sorted(files):
                if not file.startswith('.') and Path(file).suffix not in ['.html']:
                    self.explorar_arquivo(Path(root) / file)

    def gerar_relatorios(self):
        """Gera e salva os relatórios finais em JSON e Markdown na pasta /relatorios."""
        # Código para gerar relatórios permanece o mesmo, apenas o caminho de salvamento muda
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.relatorios_path / f"relatorio_exploratorio_{timestamp}.json"
        md_path = self.relatorios_path / f"resumo_executivo_{timestamp}.md"
        
        # Atualiza resumo
        resumo = self.relatorio['resumo_executivo']
        resumo['total_arquivos_processados'] = len(self.relatorio['arquivos_analisados'])
        for arq in self.relatorio['arquivos_analisados']:
            if arq['status'] == 'OK':
                cat = arq['categoria']
                if cat not in resumo['categorias']:
                    resumo['categorias'][cat] = {'arquivos': 0, 'total_registros': 0}
                resumo['categorias'][cat]['arquivos'] += 1
                resumo['categorias'][cat]['total_registros'] += arq['dimensoes']['linhas']
        
        # ... (O restante do código de geração de markdown pode ser simplificado ou mantido)
        # Salvar JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.relatorio, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Relatório JSON completo salvo em: {json_path}")
        
        # Gerar Markdown (exemplo simplificado)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# 📊 Relatório de Exploração de Dados (v3.3)\n\n")
            f.write(f"**Arquivos Processados:** {resumo['total_arquivos_processados']}\n")
            f.write(f"**Arquivos com Erro:** {resumo['total_arquivos_com_erro']}\n\n")
            for cat, info in sorted(resumo.get('categorias', {}).items()):
                 f.write(f"### Categoria: {cat}\n")
                 f.write(f"- Arquivos: {info['arquivos']} | Registros: {info['total_registros']:,}\n")

        print(f"💾 Resumo Markdown salvo em: {md_path}")


    def executar_analise_completa(self):
        print(f"🚀 INICIANDO ANÁLISE EXPLORATÓRIA v{self.relatorio['metadados']['versao'].split(' ')[0]} (ROBUSTA) 🚀")
        print("="*60)
        self.relatorio['arquivos_analisados'].clear()
        self.explorar_diretorio(self.dados_brutos)
        if self.dados_tratados.exists():
            print("\n--- Processando Dados Tratados ---\n")
            self.explorar_diretorio(self.dados_tratados)
        self.gerar_relatorios()
        print("\n" + "="*60)
        print("✅ ANÁLISE COMPLETA!")
        print("="*60)

def main():
    try:
        script_path = Path(__file__).resolve().parent
        project_base_path = script_path
        for _ in range(3):
            if (project_base_path / "dados_brutos").exists():
                break
            project_base_path = project_base_path.parent
        else:
            raise FileNotFoundError("Não foi possível localizar a pasta 'dados_brutos'.")
        print(f"Diretório do projeto detectado em: {project_base_path}")
    except Exception:
        project_base_path = Path.cwd()
        print(f"Executando no diretório de trabalho atual: {project_base_path}")
    
    if not (project_base_path / "dados_brutos").exists():
        print(f"❌ ERRO CRÍTICO: A pasta 'dados_brutos' não foi encontrada em '{project_base_path}'.")
        return

    explorer = DataExplorer(project_base_path)
    explorer.executar_analise_completa()

if __name__ == '__main__':
    main()