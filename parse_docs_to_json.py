import os
import json
import yaml
import logging
from logger_config import setup_logging
from langchain_community.document_loaders import PyMuPDFLoader

def is_table_of_contents_page(page_content: str) -> bool:
    """
    Usa uma heurística para verificar se uma página é um índice (Table of Contents).
    """
    lines_with_dots = 0
    total_lines = 0
    for line in page_content.split('\n'):
        line = line.strip()
        if not line:
            continue
        total_lines += 1
        if ". " in line and line.split(". ")[-1].strip().isdigit():
            lines_with_dots += 1
    
    if total_lines > 0 and (lines_with_dots / total_lines) > 0.3:
        return True
    return False

def parse_pdfs_to_json(data_path: str, output_path: str):
    """
    Extrai o conteúdo de todos os PDFs em um diretório, filtra páginas de índice
    e salva o resultado em um arquivo JSON estruturado.
    """
    all_pages_data = []
    logging.info("Iniciando extração de documentos com 'PyMuPDFLoader'...")
    
    for filename in os.listdir(data_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(data_path, filename)
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            
            logging.info(f"Processando arquivo '{filename}' com {len(docs)} páginas.")

            for doc in docs:
                if is_table_of_contents_page(doc.page_content):
                    logging.warning(f"Página {doc.metadata.get('page', 'N/A')} do arquivo '{filename}' ignorada (provável índice).")
                    continue

                page_data = {
                    "page_content": doc.page_content,
                    "metadata": {
                        "source": filename,
                        "page": doc.metadata.get('page', 0) + 1 # PyMuPDF começa a contar de 0
                    }
                }
                all_pages_data.append(page_data)

    logging.info(f"Extração concluída. Total de páginas válidas salvas: {len(all_pages_data)}.")

    # Salva a lista de dicionários em um arquivo JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_pages_data, f, ensure_ascii=False, indent=4)
    
    logging.info(f"Dados extraídos salvos com sucesso em: '{output_path}'")

if __name__ == '__main__':
    setup_logging()
    
    # Carrega a configuração do arquivo YAML para obter o caminho dos dados
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # Define o nome do arquivo de saída JSON
    JSON_OUTPUT_PATH = "parsed_data.json"

    parse_pdfs_to_json(
        data_path=config['data_path'],
        output_path=JSON_OUTPUT_PATH
    )