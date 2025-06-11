# scripts/00_explorador_de_dados.py
import pandas as pd
import os
import glob

# Mapeia os caminhos para suas pastas de dados
# ATENÇÃO: Verifique se este caminho base está correto para sua máquina
BASE_PATH = '/Users/louisesfer/Documents/Programming/PretaLab/pretalab-final-project'
DADOS_BRUTOS_PATH = os.path.join(BASE_PATH, 'dados_brutos')
OUTPUT_FILE = os.path.join(BASE_PATH, 'dicionario_de_dados.txt')

def explorar_arquivo(caminho_arquivo, writer):
    """Lê um arquivo (CSV ou Excel) e escreve suas informações no arquivo de saída."""
    writer.write(f"{'='*80}\n")
    writer.write(f"Analisando arquivo: {os.path.basename(caminho_arquivo)}\n")
    writer.write(f"Caminho completo: {caminho_arquivo}\n")
    writer.write(f"{'='*80}\n\n")

    try:
        if caminho_arquivo.endswith('.csv') or caminho_arquivo.endswith('.csv.gz'):
            # Tenta ler com diferentes separadores e encodings
            try:
                df = pd.read_csv(caminho_arquivo, sep=',', encoding='utf-8', on_bad_lines='warn', low_memory=False, compression='infer')
            except (UnicodeDecodeError, pd.errors.ParserError):
                try:
                    df = pd.read_csv(caminho_arquivo, sep=';', encoding='latin-1', on_bad_lines='warn', low_memory=False, compression='infer')
                except Exception as e:
                    writer.write(f"ERRO: Não foi possível ler o CSV {os.path.basename(caminho_arquivo)}. Erro: {e}\n\n")
                    return

        elif caminho_arquivo.endswith('.xlsx'):
            df = pd.read_excel(caminho_arquivo)
        else:
            writer.write(f"INFO: Arquivo com formato não suportado: {os.path.basename(caminho_arquivo)}\n\n")
            return

        writer.write("--- Primeiras 5 Linhas (Head) ---\n")
        writer.write(df.head().to_string())
        writer.write("\n\n")

        writer.write("--- Informações do DataFrame (Info) ---\n")
        # Captura a saída do .info() para escrever no arquivo
        import io
        buffer = io.StringIO()
        df.info(buf=buffer)
        writer.write(buffer.getvalue())
        writer.write("\n\n")

        writer.write("--- Resumo Estatístico (Describe) ---\n")
        writer.write(df.describe(include='all').to_string())
        writer.write("\n\n\n")
        print(f"Sucesso ao analisar: {os.path.basename(caminho_arquivo)}")

    except Exception as e:
        writer.write(f"ERRO GERAL ao processar o arquivo {os.path.basename(caminho_arquivo)}: {e}\n\n")
        print(f"Falha ao analisar: {os.path.basename(caminho_arquivo)}")

# Busca por todos os arquivos CSV, CSV.GZ e XLSX na pasta de dados brutos e subpastas
arquivos_para_analisar = glob.glob(os.path.join(DADOS_BRUTOS_PATH, '**/*.*'), recursive=True)
arquivos_para_analisar = [f for f in arquivos_para_analisar if f.endswith(('.csv', '.csv.gz', '.xlsx'))]


# Abre o arquivo de saída e começa a escrever
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write("DICIONÁRIO DE DADOS AUTOMÁTICO\n")
    f.write(f"Gerado em: {pd.to_datetime('now')}\n\n")

    for arquivo in arquivos_para_analisar:
        explorar_arquivo(arquivo, f)

print(f"\nAnálise completa. Dicionário de dados salvo em: {OUTPUT_FILE}")