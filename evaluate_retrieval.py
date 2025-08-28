import pandas as pd
import yaml
import logging
from logger_config import setup_logging
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import os
from retriever_factory import create_advanced_retriever


load_dotenv()


try:
    del os.environ['LANGCHAIN_TRACING_V2']
    del os.environ['LANGCHAIN_API_KEY']
    print("--- Logging para LangSmith DESATIVADO via código. ---")
except KeyError:
    pass


def llm_as_judge(question: str, retrieved_chunks: list, judge_model_name: str) -> dict:
    """
    Usa um LLM para julgar se o contexto recuperado é suficiente para responder à pergunta.
    """
    # Concatena o conteúdo dos chunks
    context = "\n---\n".join([chunk.page_content for chunk in retrieved_chunks])
    
    prompt_template = """
    Sua tarefa é avaliar se os Documentos de Contexto fornecidos contêm a resposta para a Pergunta do Usuário.
    Seja rigoroso: o contexto deve responder diretamente à pergunta. A simples menção de palavras-chave não é suficiente.

    Exemplo:
    Pergunta do Usuário: "Quais são as cinco fases do The OWASP Testing Framework?"
    Documentos de Contexto: "O Guia de Testes da OWASP é um recurso importante para a segurança. Ele inclui um framework de testes."
    Avaliação: false (O contexto menciona o framework, mas não lista as cinco fases.)

    ---
    Pergunta do Usuário: "{question}"

    Documentos de Contexto Recuperados:
    ---
    {context}
    ---

    Com base na sua análise, o contexto é relevante e suficiente para responder à pergunta?
    Responda APENAS com "true" ou "false".
    """
    
    prompt = PromptTemplate.from_template(prompt_template)
    try:
        #                                       ⇩⇩⇩ USE O PARÂMETRO AQUI ⇩⇩⇩
        llm = ChatOpenAI(model=judge_model_name, temperature=0)
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({
            "question": question,
            "context": context,
        })
        print(
            "================================ JUIZ EM AÇÃO ================================"
        )
        print(f"[?] PERGUNTA: {question}")
        print(f"[i] CONTEXTO FORNECIDO:\n{context}")
        print(f"[*] RESPOSTA BRUTA DO JUIZ: {response}")
        print("==============================================================================")
        is_relevant = "true" in response.lower()
        return {"is_relevant": is_relevant, "raw_response": response}
    except Exception as judge_err:
        logging.warning(
            "LLM como juiz indisponível: %s. Utilizando heurística simples de relevância.",
            judge_err,
        )
        question_terms = [t.lower() for t in question.split() if len(t) > 3]
        is_relevant = False
        for chunk in retrieved_chunks:
            content_lower = chunk.page_content.lower()
            if all(term in content_lower for term in question_terms):
                is_relevant = True
                break
        return {
            "is_relevant": is_relevant,
            "raw_response": "fallback_heuristic",
        }


def evaluate_retrieval_strategy(test_set_path: str, index_path: str, embedding_model_name: str, retriever_k: int, judge_model_name: str, retriever_config: dict):
    """
    Avalia uma estratégia de recuperação de dados usando um conjunto de testes e um juiz LLM.
    """
    logging.info(f"\n--- Avaliando a Estratégia ... com o Índice: {index_path} ---")
    test_df = pd.read_csv(test_set_path)
    
    advanced_retriever = create_advanced_retriever(
        vector_store_path=index_path,
        embedding_model_name=embedding_model_name,
        k_value=retriever_k,
        retriever_config=retriever_config
    )

    results = []
    correct_hits = 0

    for index, row in test_df.iterrows():
        question = row['pergunta']
        
        try:
            top_passages = advanced_retriever.invoke(question)
        except Exception as inv_err:
            logging.error(
                "Falha ao recuperar passagens para a pergunta '%s': %s", question, inv_err
            )
            top_passages = []

        # Usa o juiz para avaliar o contexto de alta qualidade
        #                                                         ⇩⇩⇩ PASSE O PARÂMETRO AQUI ⇩⇩⇩
        judgement = llm_as_judge(question, top_passages, judge_model_name)
        
        if judgement["is_relevant"]:
            correct_hits += 1
        
        logging.info(f"Pergunta: {question[:50]}... | Relevante: {judgement['is_relevant']}")
        results.append(judgement)

    if len(test_df) > 0:
        accuracy = (correct_hits / len(test_df)) * 100
    else:
        accuracy = 0
        
    logging.info(f"--- Resultado Final para '{index_path}' (Busca Híbrida + Re-ranker) ---")
    logging.info(f"Taxa de Acerto da Recuperação: {accuracy:.2f}% ({correct_hits}/{len(test_df)})")
    return accuracy


if __name__ == '__main__':
    setup_logging()
    
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    all_results = []
    for strategy in config['ingestion_strategies']:
        strategy_id = strategy['id']
        index_path = config['vector_store_path_template'].format(id=strategy_id)

        try:
            accuracy = evaluate_retrieval_strategy(
                test_set_path=config['test_set_path'],
                index_path=index_path,
                embedding_model_name=strategy['embedding_model'],
                retriever_k=config['evaluator']['retriever_k'],
                judge_model_name=config['evaluator']['llm_judge'],
                retriever_config=config['retriever_models']
            )
        except Exception as eval_err:
            logging.error(
                "Erro ao avaliar a estratégia %s: %s", strategy_id, eval_err
            )
            accuracy = 0.0

        chunk_size = strategy.get('chunk_size', 'N/A') 

        all_results.append({
            'strategy_id': strategy_id, 
            'chunk_size': chunk_size,
            'embedding_model': strategy['embedding_model'],
            'accuracy': f"{accuracy:.2f}%"
        })

    results_df = pd.DataFrame(all_results)
    results_path = config['results_path']
    results_df.to_csv(results_path, index=False)