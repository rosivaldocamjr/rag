import os
import json
import yaml
import logging
from logger_config import setup_logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# Removido: from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from pymilvus import connections, Collection, utility, Partition

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

from dotenv import load_dotenv
load_dotenv()

MILVUS_URI = os.getenv("MILVUS_AMB_URI")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME")
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME")


def insert_data_into_milvus(collection: Collection, chunks: list, embedding_model, partition_name: str):    
    """
    Gera embeddings para os chunks e os insere na coleção do Milvus.
    """
    logging.info(f"Iniciando a inserção de {len(chunks)} chunks na partição '{partition_name}'...")

    texts = [chunk.page_content for chunk in chunks]
    try:
        embeddings = embedding_model.embed_documents(texts)
        logging.info("Embeddings gerados com sucesso.")
    except Exception as e:
        logging.error(f"Falha ao gerar embeddings: {e}")
        return
    
    entities = []
    for i, chunk in enumerate(chunks):
        entity = {
            "embedding": embeddings[i],
            "chunk_text": chunk.page_content,
            "source": chunk.metadata.get("source", "N/A"),
            "page": int(chunk.metadata.get("page", 0))
        }
        entities.append(entity)
        
    try:
        # Insere os dados na coleção
        collection.insert(entities, partition_name=partition_name)
        # Realiza o "flush" para garantir que os dados sejam escritos no disco
        collection.flush()
        logging.info(f"{len(entities)} chunks inseridos com sucesso na partição '{partition_name}'.")

    except Exception as e:
        logging.error(f"Erro ao inserir dados no Milvus: {e}")


def process_and_store_documents(docs: list, strategy: dict):
    """
    Recebe uma lista de documentos, aplica uma estratégia de chunking
    e armazena o resultado no Milvus.
    """
    chunk_method = strategy.get("chunk_method", "recursive")
    embedding_model_name = strategy['embedding_model']
    partition_name = strategy['partition_name']

    logging.info(f"\n--- Processando com: chunk_method={chunk_method}, model='{embedding_model_name}', partition='{partition_name}' ---")

    # Carrega o modelo de embedding
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

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

    try:
        connections.connect(alias="default", uri=MILVUS_URI, db_name=MILVUS_DB_NAME)
        logging.info(f"Conexão com Milvus estabelecida em '{MILVUS_URI}' no DB '{MILVUS_DB_NAME}'.")

        if not utility.has_collection(MILVUS_COLLECTION_NAME):
            logging.error(f"A coleção '{MILVUS_COLLECTION_NAME}' não foi encontrada no Milvus. Crie-a primeiro.")
            return

        collection = Collection(name=MILVUS_COLLECTION_NAME)
        
        if collection.has_partition(partition_name):
            logging.warning(f"Partição '{partition_name}' já existe. Removendo dados antigos...")
            collection.drop_partition(partition_name)
       
        logging.info(f"Criando nova partição: '{partition_name}'")
        collection.create_partition(partition_name)

        collection.load()

        insert_data_into_milvus(collection, chunks, embedding_model, partition_name)
        
    except Exception as e:
        logging.error(f"Ocorreu um erro durante a operação com o Milvus: {e}")
    finally:
        connections.disconnect(alias="default")
        logging.info("Conexão com Milvus encerrada.")


if __name__ == '__main__':
    setup_logging()
    
    JSON_PATH = "parsed_data.json"
    
    if not os.path.exists(JSON_PATH):
        logging.error(f"Arquivo '{JSON_PATH}' não encontrado. Execute 'parse_docs_to_json.py' primeiro.")
        exit()

    logging.info(f"Carregando documentos de '{JSON_PATH}'...")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = [Document(page_content=item['page_content'], metadata=item['metadata']) for item in data]
    logging.info(f"Total de {len(documents)} documentos carregados.")

    for strategy in config['ingestion_strategies']:
        strategy_id = strategy['id']
        logging.info(f"\n{'='*20} PROCESSANDO ESTRATÉGIA {strategy_id} {'='*20}")

        process_and_store_documents(
            docs=documents,
            strategy=strategy
        )