import os
import json
import yaml
import logging
from logger_config import setup_logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document 

def create_vector_store(docs: list, strategy: dict, index_path: str):
    """
    Recebe uma lista de documentos (já carregados), aplica uma estratégia de chunking
    e cria um banco de vetores FAISS.
    """
    chunk_method = strategy.get("chunk_method", "recursive")
    embedding_model_name = strategy['embedding_model']

    logging.info(f"\n--- Criando Vector Store com a estratégia: chunk_method={chunk_method}, model='{embedding_model_name}' ---")

    # Carrega o modelo de embedding
    embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model_name)

    # Escolhe o método de chunking com base na estratégia
    if chunk_method == "semantic":
        text_splitter = SemanticChunker(embedding_model)
    else: # O padrão será "recursive"
        chunk_size = strategy.get("chunk_size", 1000)
        chunk_overlap = strategy.get("chunk_overlap", 200)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    chunks = text_splitter.split_documents(docs)
    logging.info(f"Total de chunks gerados: {len(chunks)}")

    logging.info("Gerando embeddings e criando o banco de vetores...")
    vector_store = FAISS.from_documents(chunks, embedding_model)
    logging.info("Banco de vetores criado com sucesso.")

    # Certifique-se de que o diretório de destino exista antes de salvar
    os.makedirs(index_path, exist_ok=True)
    vector_store.save_local(index_path)
    logging.info(f"Índice salvo em: {index_path}")


if __name__ == '__main__':
    setup_logging()

    # Carrega as configurações do projeto
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # 1. Define o caminho para o arquivo JSON pré-processado
    JSON_PATH = "parsed_data.json"
    
    # Verifica se o arquivo JSON existe antes de continuar
    if not os.path.exists(JSON_PATH):
        logging.error(f"Arquivo '{JSON_PATH}' não encontrado. Execute o script 'parse_docs_to_json.py' primeiro.")
        exit() # Encerra o script se o arquivo não existir

    # 2. Carrega os dados extraídos do arquivo JSON
    logging.info(f"Carregando documentos pré-processados de '{JSON_PATH}'...")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 3. Converte os dados do JSON de volta para objetos Document do LangChain
    # Isso é necessário para que as funções do LangChain (como o text_splitter) funcionem corretamente.
    documents = [Document(page_content=item['page_content'], metadata=item['metadata']) for item in data]
    logging.info(f"Total de {len(documents)} documentos carregados do JSON.")

    # 4. Itera sobre as estratégias para criar os vector stores
    for strategy in config['ingestion_strategies']:
        strategy_id = strategy['id']
        index_path = config['vector_store_path_template'].format(id=strategy_id)

        logging.info(f"\n{'='*20} PROCESSANDO ESTRATÉGIA {strategy_id} {'='*20}")

        # Opcional: Verifica se o índice já existe para não reprocessar
        if os.path.exists(index_path):
            logging.info(f"Índice para a estratégia {strategy_id} já existe em '{index_path}'. Pulando.")
            continue

        # Chama a função de criação do vector store com os documentos carregados
        create_vector_store(
            docs=documents,
            strategy=strategy,
            index_path=index_path
        )