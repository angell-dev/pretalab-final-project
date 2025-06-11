#!/usr/bin/env python3
"""
Script para baixar arquivos do Disque 100 diretamente das URLs fornecidas
URLs base: https://dadosabertos.mdh.gov.br/
Verifica se arquivos já existem antes de baixar novamente
"""

import requests
import os
import time
from datetime import datetime

# Configurações
OUTPUT_DIR = "dados_brutos/disque100"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
MIN_FILE_SIZE = 1024  # Tamanho mínimo em bytes para considerar um arquivo válido (1KB)

# URLs diretas dos arquivos
URLS_DISQUE100 = [
    ("2020_1", "Primeiro semestre de 2020", "https://dadosabertos.mdh.gov.br/disque100-primeiro-semestre-2020.csv"),
    ("2020_2", "Segundo semestre de 2020", "https://dadosabertos.mdh.gov.br/disque100-segundo-semestre-2020.csv"),
    ("2021_1", "Primeiro semestre de 2021", "https://dadosabertos.mdh.gov.br/disque100-primeiro-semestre-2021.csv"),
    ("2021_2", "Segundo semestre de 2021", "https://dadosabertos.mdh.gov.br/disque100-segundo-semestre-2021.csv"),
    ("2022_1", "Primeiro semestre de 2022", "https://dadosabertos.mdh.gov.br/disque100-primeiro-semestre-2022.csv"),
    ("2022_2", "Segundo semestre de 2022", "https://dadosabertos.mdh.gov.br/disque100-segundo-semestre-2022.csv"),
    ("2023_1", "Primeiro semestre de 2023", "https://dadosabertos.mdh.gov.br/disque100-primeiro-semestre-2023.csv"),
    ("2023_2", "Segundo semestre de 2023", "https://dadosabertos.mdh.gov.br/disque100-segundo-semestre-2023.csv"),
    ("2024_1", "Primeiro semestre de 2024", "https://dadosabertos.mdh.gov.br/disque100-primeiro-semestre-2024.csv"),
    ("2024_2", "Segundo semestre de 2024", "https://dadosabertos.mdh.gov.br/disque100-segundo-semestre-2024.csv"),
    ("2025_1", "Primeiro trimestre de 2025", "https://dadosabertos.mdh.gov.br/disque100-primeiro-trimestre-2025.csv"),
]

# Criar diretório de saída
os.makedirs(OUTPUT_DIR, exist_ok=True)

def formatar_tamanho(bytes):
    """Formata tamanho em bytes para formato legível"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} TB"

def verificar_arquivo_existente(caminho_arquivo):
    """
    Verifica se um arquivo existe e é válido
    Retorna: (existe, é_valido, tamanho, mensagem)
    """
    if not os.path.exists(caminho_arquivo):
        return False, False, 0, "Arquivo não existe"
    
    try:
        tamanho = os.path.getsize(caminho_arquivo)
        
        # Verificar tamanho mínimo
        if tamanho < MIN_FILE_SIZE:
            return True, False, tamanho, f"Arquivo muito pequeno ({formatar_tamanho(tamanho)})"
        
        # Verificar se é um arquivo CSV válido (não HTML)
        with open(caminho_arquivo, 'r', encoding='utf-8', errors='ignore') as f:
            # Ler as primeiras linhas para verificar
            primeiras_linhas = []
            for i in range(min(5, tamanho // 100)):  # Ler até 5 linhas
                linha = f.readline()
                if not linha:
                    break
                primeiras_linhas.append(linha.lower())
            
            # Verificar se é HTML
            conteudo_inicio = ''.join(primeiras_linhas)
            if any(tag in conteudo_inicio for tag in ['<!doctype', '<html', '<head', '<body']):
                return True, False, tamanho, "Arquivo é HTML, não CSV"
            
            # Verificar se tem conteúdo que parece CSV (vírgulas ou ponto-e-vírgula)
            if not any(char in conteudo_inicio for char in [',', ';']):
                return True, False, tamanho, "Arquivo não parece ser CSV válido"
        
        return True, True, tamanho, f"Arquivo válido ({formatar_tamanho(tamanho)})"
        
    except Exception as e:
        return True, False, 0, f"Erro ao verificar arquivo: {str(e)}"

def baixar_arquivo(url, nome_arquivo, max_tentativas=3, forcar_download=False):
    """Baixa um arquivo com retry e mostra progresso"""
    caminho_completo = os.path.join(OUTPUT_DIR, nome_arquivo)
    
    # Verificar se arquivo já existe
    if not forcar_download:
        existe, valido, tamanho, mensagem = verificar_arquivo_existente(caminho_completo)
        if existe and valido:
            print(f"  ✓ Arquivo já existe e é válido: {mensagem}")
            print(f"  ⏭️  Pulando download")
            return True
        elif existe and not valido:
            print(f"  ⚠️  Arquivo existe mas não é válido: {mensagem}")
            print(f"  🔄 Tentando baixar novamente...")
    
    for tentativa in range(max_tentativas):
        try:
            print(f"  Tentativa {tentativa + 1}/{max_tentativas}")
            
            # Headers para parecer um navegador real
            headers = {
                'User-Agent': USER_AGENT,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0'
            }
            
            # Fazer requisição com timeout maior
            response = requests.get(url, headers=headers, stream=True, timeout=60, verify=True)
            
            # Verificar status
            if response.status_code != 200:
                print(f"    ✗ Status HTTP: {response.status_code}")
                if tentativa < max_tentativas - 1:
                    time.sleep(3)
                continue
            
            # Verificar se é CSV
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/csv' not in content_type and 'application/csv' not in content_type:
                # Tentar baixar mesmo assim, pode ser que o content-type esteja errado
                print(f"    ⚠ Content-Type não é CSV: {content_type}")
            
            # Obter tamanho do arquivo
            tamanho_total = int(response.headers.get('Content-Length', 0))
            
            # Iniciar contagem de tempo
            start_time = time.time()
            
            # Baixar arquivo
            with open(caminho_completo, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Mostrar progresso
                        if tamanho_total > 0:
                            progresso = (downloaded / tamanho_total) * 100
                            print(f"\r    Progresso: {progresso:.1f}% ({formatar_tamanho(downloaded)}/{formatar_tamanho(tamanho_total)})", end='', flush=True)
                        else:
                            print(f"\r    Baixado: {formatar_tamanho(downloaded)}", end='', flush=True)
            
            # Calcular tempo decorrido
            elapsed_time = time.time() - start_time
            print()  # Nova linha após progresso
            
            # Verificar se o arquivo foi baixado corretamente
            existe, valido, tamanho, mensagem = verificar_arquivo_existente(caminho_completo)
            if existe and valido:
                print(f"    ✓ Download concluído! {mensagem}")
                print(f"    ⏱️  Tempo de download: {elapsed_time:.2f} segundos")
                # Calcular velocidade média se o tamanho total estiver disponível
                if tamanho_total > 0:
                    speed = tamanho_total / elapsed_time  # bytes por segundo
                    print(f"    🚀 Velocidade média: {formatar_tamanho(speed)}/s")
                return True
            else:
                print(f"    ✗ Download falhou: {mensagem}")
                if existe:
                    os.remove(caminho_completo)
            
        except requests.exceptions.ConnectionError:
            print(f"    ✗ Erro de conexão")
        except requests.exceptions.Timeout:
            print(f"    ✗ Timeout na requisição")
        except requests.exceptions.RequestException as e:
            print(f"    ✗ Erro na requisição: {str(e)}")
        except Exception as e:
            print(f"    ✗ Erro inesperado: {str(e)}")
        
        # Aguardar antes de tentar novamente
        if tentativa < max_tentativas - 1:
            print(f"    Aguardando 5 segundos antes de tentar novamente...")
            time.sleep(5)
    
    return False

def verificar_conectividade():
    """Verifica se há conectividade com o servidor"""
    try:
        print("Verificando conectividade com dadosabertos.mdh.gov.br...")
        response = requests.get("https://dadosabertos.mdh.gov.br", timeout=10)
        if response.status_code == 200:
            print("✓ Servidor acessível\n")
            return True
        else:
            print(f"⚠ Servidor retornou status: {response.status_code}\n")
            return True  # Tentar mesmo assim
    except:
        print("✗ Não foi possível acessar o servidor\n")
        return False

def listar_arquivos_existentes():
    """Lista arquivos já baixados no diretório"""
    print(f"\nVerificando arquivos existentes em {OUTPUT_DIR}...")
    arquivos_existentes = []
    
    try:
        arquivos_csv = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv')]
        if arquivos_csv:
            print(f"Encontrados {len(arquivos_csv)} arquivos CSV:")
            for arquivo in sorted(arquivos_csv):
                caminho = os.path.join(OUTPUT_DIR, arquivo)
                existe, valido, tamanho, mensagem = verificar_arquivo_existente(caminho)
                status = "✓" if valido else "⚠"
                print(f"  {status} {arquivo} - {mensagem}")
                if valido:
                    arquivos_existentes.append(arquivo)
        else:
            print("  Nenhum arquivo CSV encontrado")
    except Exception as e:
        print(f"  Erro ao listar arquivos: {str(e)}")
    
    return arquivos_existentes

def main(forcar_download=False):
    print("=== Baixador Direto de URLs - Disque 100 ===")
    print(f"Diretório de saída: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Hora de início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if forcar_download:
        print("\n⚠️  MODO FORÇAR DOWNLOAD ATIVADO - Todos os arquivos serão baixados novamente!")
    
    # Listar arquivos existentes
    arquivos_existentes = listar_arquivos_existentes()
    
    # Verificar conectividade
    print()
    if not verificar_conectividade():
        print("Continuando mesmo assim...\n")
    
    print(f"Processando {len(URLS_DISQUE100)} arquivos do Disque 100...\n")
    
    arquivos_baixados = []
    arquivos_pulados = []
    arquivos_falhados = []
    
    for i, (periodo, descricao, url) in enumerate(URLS_DISQUE100, 1):
        print(f"[{i}/{len(URLS_DISQUE100)}] {descricao}")
        print(f"  Período: {periodo}")
        
        # Nome do arquivo local
        nome_arquivo = f"disque100_{periodo}.csv"
        print(f"  Arquivo: {nome_arquivo}")
        
        # Verificar se já existe
        caminho_completo = os.path.join(OUTPUT_DIR, nome_arquivo)
        if not forcar_download:
            existe, valido, tamanho, mensagem = verificar_arquivo_existente(caminho_completo)
            if existe and valido:
                print(f"  ✓ {mensagem}")
                print(f"  ⏭️  Pulando download (arquivo já existe)\n")
                arquivos_pulados.append(descricao)
                continue
        
        print(f"  URL: {url}")
        
        # Tentar baixar
        if baixar_arquivo(url, nome_arquivo, forcar_download=forcar_download):
            arquivos_baixados.append(descricao)
            print(f"  ✓ SUCESSO\n")
        else:
            arquivos_falhados.append(descricao)
            print(f"  ✗ FALHA\n")
        
        # Pequena pausa entre downloads
        if i < len(URLS_DISQUE100):
            time.sleep(2)
    
    # Resumo final
    print("\n" + "="*60)
    print("RESUMO DO PROCESSAMENTO")
    print("="*60)
    print(f"Total de arquivos: {len(URLS_DISQUE100)}")
    print(f"⏭️  Pulados (já existem): {len(arquivos_pulados)}")
    print(f"✓ Baixados com sucesso: {len(arquivos_baixados)}")
    print(f"✗ Falharam: {len(arquivos_falhados)}")
    
    if arquivos_pulados:
        print("\nArquivos que já existiam:")
        for arquivo in arquivos_pulados:
            print(f"  ⏭️  {arquivo}")
    
    if arquivos_baixados:
        print("\nArquivos baixados com sucesso:")
        for arquivo in arquivos_baixados:
            print(f"  ✓ {arquivo}")
    
    if arquivos_falhados:
        print("\nArquivos que falharam:")
        for arquivo in arquivos_falhados:
            print(f"  ✗ {arquivo}")
    
    # Listar todos os arquivos no diretório
    print(f"\nTodos os arquivos em {OUTPUT_DIR}:")
    try:
        arquivos_csv = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv')]
        if arquivos_csv:
            total_tamanho = 0
            for arquivo in sorted(arquivos_csv):
                caminho = os.path.join(OUTPUT_DIR, arquivo)
                existe, valido, tamanho, mensagem = verificar_arquivo_existente(caminho)
                status = "✓" if valido else "⚠"
                print(f"  {status} {arquivo} - {mensagem}")
                if valido:
                    total_tamanho += tamanho
            print(f"\nTamanho total dos arquivos válidos: {formatar_tamanho(total_tamanho)}")
        else:
            print("  Nenhum arquivo CSV encontrado")
    except:
        print("  Erro ao listar arquivos")
    
    print(f"\nHora de término: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    import sys
    
    # Verificar se foi passado o argumento --force
    forcar = "--force" in sys.argv or "-f" in sys.argv
    
    if len(sys.argv) > 1 and sys.argv[1] not in ["--force", "-f"]:
        print("Uso: python script.py [--force|-f]")
        print("  --force, -f  Força o download mesmo se os arquivos já existirem")
        sys.exit(1)
    
    main(forcar_download=forcar)