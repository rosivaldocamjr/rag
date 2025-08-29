import logging
import os
from langchain.schema.retriever import BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
except Exception as e:
    HuggingFaceCrossEncoder = None
    logging.warning(
        "Falha ao importar HuggingFaceCrossEncoder. O re-ranking será desativado. Erro: %s", e
    )

def create_advanced_retriever(
    vector_store_path: str,
    embedding_model_name: str,
    k_value: int,
    retriever_config: dict,
) -> BaseRetriever:
    """
    Cria e configura um retriever avançado que utiliza busca híbrida e re-ranking.

    Este método é controlado por um dicionário de configuração para carregar os
    modelos de fallback e de re-ranking.

    Args:
        vector_store_path (str): Caminho para o diretório do índice FAISS.
        embedding_model_name (str): Nome ou caminho do modelo de embeddings principal.
        k_value (int): Quantidade de documentos a retornar após o re-ranking.
        retriever_config (dict): Dicionário com configurações dos modelos.
            Exemplo:
            {
                "default_embedding_fallback": "all-MiniLM-L6-v2",
                "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
            }

    Returns:
        BaseRetriever: Um objeto retriever pronto para ser usado (pode ser
        ContextualCompressionRetriever ou EnsembleRetriever).
    """
    logging.info("Criando o retriever avançado a partir da configuração...")

    # --- 1. Carrega o modelo de embeddings a partir da configuração ---
    fallback_model = retriever_config.get(
        "default_embedding_fallback", "all-MiniLM-L6-v2"
    )
    
    resolved_embedding_name = embedding_model_name
    if os.path.sep in embedding_model_name and not os.path.exists(
        embedding_model_name
    ):
        logging.warning(
            "Modelo de embeddings '%s' não encontrado localmente. Usando fallback '%s'.",
            embedding_model_name,
            fallback_model,
        )
        resolved_embedding_name = fallback_model

    try:
        embedding_model = HuggingFaceEmbeddings(model_name=resolved_embedding_name)
    except Exception as embed_err:
        logging.error(
            "Falha ao carregar o modelo de embeddings '%s': %s. Usando fallback '%s'.",
            resolved_embedding_name,
            embed_err,
            fallback_model,
        )
        embedding_model = HuggingFaceEmbeddings(model_name=fallback_model)

    # --- 2. Carrega o índice FAISS ---
    if not os.path.exists(vector_store_path):
        raise FileNotFoundError(
            f"O caminho do índice FAISS '{vector_store_path}' não existe. Execute o script de ingestão."
        )
    vector_store = FAISS.load_local(
        vector_store_path,
        embedding_model,
        # AVISO DE SEGURANÇA: Esta flag é necessária, mas garanta que os índices 
        # sejam sempre gerados por você em um ambiente seguro.
        allow_dangerous_deserialization=True,
    )

    # --- 3. Prepara os retrievers base ---
    all_chunks = list(vector_store.docstore._dict.values())
    if not all_chunks:
        raise ValueError(
            "Nenhum documento encontrado no docstore do FAISS. O índice pode estar vazio."
        )

    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 15})
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = 15

    # --- 4. Cria o retriever híbrido ---
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.25, 0.75],
    )

    # --- 5. Configura o re-ranking a partir da configuração ---
    if HuggingFaceCrossEncoder is None:
        logging.warning(
            "Biblioteca de re-ranking indisponível; retornando retriever híbrido sem re-ranking."
        )
        return ensemble_retriever

    try:
        reranker_model_name = retriever_config.get(
            "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        logging.info("Carregando modelo de re-ranking: '%s'", reranker_model_name)
        
        re_ranker_model = HuggingFaceCrossEncoder(model_name=reranker_model_name)
        compressor = CrossEncoderReranker(model=re_ranker_model, top_n=k_value)
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever,
        )
        logging.info("Retriever avançado criado com sucesso (Híbrido + Re-ranker).")
        return compression_retriever
        
    except Exception as rerank_err:
        logging.warning(
            "Falha ao configurar o re-ranker CrossEncoder: %s. Retornando retriever híbrido.",
            rerank_err,
        )
        return ensemble_retriever