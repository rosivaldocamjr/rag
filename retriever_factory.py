import logging
import os
from langchain.schema.retriever import BaseRetriever
# Removido: from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_milvus.vectorstores import Milvus
from pymilvus import connections, utility, Collection, Partition
from dotenv import load_dotenv

load_dotenv()

try:
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
except Exception as e:
    HuggingFaceCrossEncoder = None
    logging.warning(
        "Falha ao importar HuggingFaceCrossEncoder. O re-ranking será desativado. Erro: %s", e
    )

def get_all_documents_from_milvus(collection: Collection, partition_name: str) -> list[Document]:
    """
    Consulta a coleção Milvus para recuperar todos os documentos armazenados.
    Isso é necessário para inicializar o retriever BM25.
    """
    
    logging.info(f"Consultando todos os documentos da partição '{partition_name}' para o BM25...")
    
    collection.load([partition_name])
    
    res = collection.query(
        expr="", 
        output_fields=["chunk_text", "source", "page"], 
        partition_names=[partition_name],
        limit=16384
    )
    
    docs = []
    for hit in res:
        doc = Document(
            page_content=hit['chunk_text'],
            metadata={
                'source': hit['source'],
                'page': hit['page']
            }
        )
        docs.append(doc)
    
    logging.info(f"{len(docs)} documentos recuperados da partição '{partition_name}' para o BM25.")
    
    return docs


def create_advanced_retriever(
    partition_name: str, 
    embedding_model_name: str,
    k_value: int,
    retriever_config: dict,
) -> BaseRetriever:
    """
    Cria e configura um retriever avançado que utiliza busca híbrida (Milvus + BM25) e re-ranking.
    """
    logging.info(f"Criando retriever avançado para a partição '{partition_name}'...")

    # --- 1. Carrega o modelo de embeddings (semelhante ao anterior) ---
    fallback_model = retriever_config.get(
        "default_embedding_fallback", "all-MiniLM-L6-v2"
    )
    # A lógica de fallback do modelo de embedding permanece a mesma
    resolved_embedding_name = embedding_model_name
    if os.path.sep in embedding_model_name and not os.path.exists(embedding_model_name):
        logging.warning(
            "Modelo de embeddings '%s' não encontrado. Usando fallback '%s'.",
            embedding_model_name, fallback_model
        )
        resolved_embedding_name = fallback_model
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=resolved_embedding_name)
    except Exception as embed_err:
        logging.error(
            "Falha ao carregar embeddings '%s': %s. Usando fallback '%s'.",
            resolved_embedding_name, embed_err, fallback_model
        )
        embedding_model = HuggingFaceEmbeddings(model_name=fallback_model)

    # --- 2. Conecta ao Milvus e prepara os retrievers ---
    try:
        # Pega as informações de conexão do .env
        uri = os.getenv("MILVUS_AMB_URI")
        db_name = os.getenv("MILVUS_DB_NAME")
        collection_name = os.getenv("MILVUS_COLLECTION_NAME")
        
        connections.connect(alias="default", uri=uri, db_name=db_name)
        logging.info(f"Conexão com Milvus estabelecida em '{uri}'.")
        
        if not utility.has_collection(collection_name):
            raise FileNotFoundError(f"A coleção '{collection_name}' não existe no Milvus. Execute o script de ingestão.")

        milvus_retriever = Milvus(
            embedding_function=embedding_model,
            collection_name=collection_name,
            connection_args={"alias": "default"},
            auto_id=True, 
            text_field="chunk_text", 
            vector_field="embedding" 
        ).as_retriever(search_kwargs={"k": 15})
        
        logging.info("Retriever do Milvus (semântico) criado com sucesso.")

        milvus_collection = Collection(name=collection_name)
        milvus_collection.load()
        all_chunks = get_all_documents_from_milvus(milvus_collection, partition_name)
        
        if not all_chunks:
            raise ValueError("Nenhum documento encontrado no Milvus para inicializar o BM25.")

        bm25_retriever = BM25Retriever.from_documents(all_chunks)
        bm25_retriever.k = 15
        logging.info("Retriever BM25 (palavra-chave) criado com sucesso.")

    except Exception as e:
        logging.error(f"Falha ao conectar ou configurar retrievers com o Milvus: {e}")
        raise

    # --- 4. Cria o retriever híbrido (Ensemble) ---
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, milvus_retriever],
        weights=[0.25, 0.75], # Dando mais peso para a busca semântica
    )

    # --- 5. Configura o re-ranking (inalterado) ---
    if HuggingFaceCrossEncoder is None:
        logging.warning("Retornando retriever híbrido sem re-ranking.")
        return ensemble_retriever

    try:
        reranker_model_name = retriever_config.get("reranker_model")
        logging.info("Carregando modelo de re-ranking: '%s'", reranker_model_name)
        
        re_ranker_model = HuggingFaceCrossEncoder(model_name=reranker_model_name)
        compressor = CrossEncoderReranker(model=re_ranker_model, top_n=k_value)
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever,
        )
        logging.info("Retriever avançado criado com sucesso (Híbrido Milvus + Re-ranker).")
        return compression_retriever
        
    except Exception as rerank_err:
        logging.warning(
            "Falha ao configurar o re-ranker: %s. Retornando retriever híbrido.", rerank_err
        )
        return ensemble_retriever