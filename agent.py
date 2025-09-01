import logging
from logger_config import setup_logging
from langchain_openai import ChatOpenAI
from langchain.agents import tool, AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import yaml
from retriever_factory import create_advanced_retriever

load_dotenv()

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

strategy_id_to_use = config['agent']['strategy_to_use']

chosen_strategy = next(
    (s for s in config['ingestion_strategies'] if s['id'] == strategy_id_to_use),
    None,
)

if not chosen_strategy:
    raise ValueError(
        f"Estratégia com id '{strategy_id_to_use}' não encontrada no config.yaml"
    )

logging.info(
    f"Agente será executado com a Estratégia {chosen_strategy['id']}"
)

BEST_EMBEDDING_MODEL = chosen_strategy['embedding_model']
PARTITION_TO_USE = chosen_strategy['partition_name']

try:
    retriever = create_advanced_retriever(
        partition_name=PARTITION_TO_USE,
        embedding_model_name=BEST_EMBEDDING_MODEL,
        k_value=config['agent']['retriever_k'],
        retriever_config=config['retriever_models'],
    )
except Exception as retr_err:  
    logging.error(
        "Falha ao criar o retriever para o agente: %s", retr_err
    )
    retriever = None

@tool
def search_in_documents(search_query: str) -> str: 
    """
    Realiza uma busca semântica no OWASP Application Security Verification Standard v5.0.0 para encontrar 
    requisitos e orientações sobre segurança de aplicações e serviços web. A entrada deve ser uma pergunta 
    clara ou palavras-chave. Este é o recurso principal para obter o contexto necessário para responder a 
    dúvidas relacionadas a práticas, controles e verificações de segurança em software, cobrindo desde 
    requisitos básicos até mecanismos avançados, organizados em níveis e capítulos.
    """

    logging.info(f"--- Agente chamou a ferramenta com query: '{search_query}' ---") # 

    if retriever is None:
        return (
            "O mecanismo de busca não está disponível. Verifique se o índice de vetores foi gerado "
            "(execute o script de ingestão) ou se há dependências ausentes."
        )

    docs = retriever.invoke(search_query)
    if not docs:
        return "Nenhuma informação relevante foi encontrada nos documentos para esta consulta."

    context = "\n\n---\n\n".join(
        [
            f"Fonte: {doc.metadata.get('source', 'N/A')}, Página: {doc.metadata.get('page', 'N/A')}\nConteúdo: {doc.page_content}"
            for doc in docs
        ]
    )
    return context

def create_rag_agent():
    
    tools = [search_in_documents]
    
    SYSTEM_PROMPT = """
    Você é um assistente especialista em análise de segurança de aplicações web chamado AnalistaIA. 
    Sua personalidade é analítica, precisa e objetiva.

    Siga estritamente estas regras:

    1. Raciocínio (Chain of Thought): Antes de responder, pense passo a passo. Primeiro, decomponha a pergunta 
    do usuário em palavras-chave ou questões específicas. Segundo, use a ferramenta search_in_documents para 
    cada questão específica. Terceiro, sintetize as informações recuperadas em uma resposta coesa.

    2. Uso da Ferramenta: Sempre utilize a ferramenta search_in_documents para obter o contexto. Nunca responda 
    com base em conhecimento prévio. Se a primeira busca não retornar resultados, tente reformular a query 
    de busca para ser mais específica ou mais geral, dependendo do caso.

    3. Formato da Resposta: A resposta final DEVE seguir este formato:

    - Resposta Direta: Responda à pergunta de forma clara e direta.
    - Evidências: Apresente os trechos exatos dos documentos que suportam sua resposta, em formato de citação.
    - Fontes: Liste todas as fontes consultadas no formato: (Arquivo: [nome_do_arquivo], Página: [numero_da_pagina]).

    4. Incerteza: Se as informações nos documentos não forem suficientes para responder à pergunta, afirme 
    claramente: Com base nos documentos fornecidos, não foi possível encontrar uma resposta para esta pergunta.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    llm = ChatOpenAI(model=config['agent']['agent_llm'], temperature=0)
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor

if __name__ == '__main__':

    setup_logging()
    
    rag_agent = create_rag_agent()
    logging.info("Agente RAG iniciado. Faça suas perguntas. Pressione Ctrl+C para sair.")

    try:
        while True:
            
            question = input("\nSua Pergunta: ")

            response = rag_agent.invoke({"input": question})

            logging.info("\n--- Resposta do Agente ---")
            logging.info(response["output"])
    except KeyboardInterrupt:
        
        logging.info("\n\nEncerrando o agente. Até logo!")