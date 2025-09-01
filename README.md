# RAG com Busca Híbrida, Re-ranking e LLM-as-Judge

Este projeto implementa um sistema avançado de RAG (Retrieval-Augmented Generation) projetado para responder perguntas com base em uma coleção de documentos privados em formato PDF. Ele utiliza técnicas sofisticadas para garantir que as respostas sejam precisas e relevantes, buscando o contexto em um banco de dados vetorial Milvus.

---

### Core Features
- Busca Híbrida: Combina a busca por palavras-chave (BM25) com a busca por similaridade semântica (vetores de embedding), garantindo a captura tanto de termos exatos quanto do significado contextual.

- Re-ranking de Resultados: Utiliza um modelo Cross-Encoder para reordenar os resultados da busca híbrida, trazendo os trechos mais relevantes para o topo antes de enviá-los ao LLM.

- Banco de Dados Vetorial Milvus: Armazena os vetores de embeddings para buscas semânticas rápidas e escaláveis, com suporte a partições para isolar diferentes estratégias de ingestão.

- Agente Inteligente (LangChain): Um agente conversacional que utiliza as ferramentas de busca para raciocinar sobre a pergunta do usuário e formular respostas detalhadas com base nas fontes encontradas.

- Pipeline de Avaliação: Inclui um módulo para avaliar objetivamente a qualidade do sistema de recuperação de informações usando um LLM como "juiz" (LLM-as-a-Judge), gerando uma métrica de acurácia.

- Estratégias de Ingestão Configuráveis: Permite testar diferentes métodos de "chunking" (divisão de texto) e modelos de embedding através de um único arquivo de configuração (config.yaml).

---

### Arquitetura do Projeto
/<br>
├── data/                     # Pasta para colocar os documentos PDF de entrada<br>
├── local_models/             # (Opcional) Pasta para modelos de embedding locais<br>
├── agent.py                  # Script para iniciar e interagir com o agente RAG<br>
├── evaluate_retrieval.py     # Script para rodar a avaliação de performance do retriever<br>
├── ingestion.py              # Script para processar PDFs e carregar os dados no Milvus<br>
├── logger_config.py          # Configuração centralizada de logs do projeto<br>
├── parse_docs_to_json.py     # Script auxiliar para extrair texto dos PDFs<br>
├── retriever_factory.py      # Módulo central que constrói o retriever avançado<br>
├── config.yaml               # Arquivo de configuração central para todo o projeto<br>
├── evaluation_results.csv    # Resultados das avaliações do retriever<br>
├── parsed_data.json          # Dados já processados e normalizados<br>
├── test_set.csv              # Dataset de teste para avaliação do sistema<br>
├── requirements.in           # Lista mínima de dependências (antes do pip-compile)<br>
├── requirements.txt          # Dependências completas e compiladas do projeto<br>
├── README.md                 # Documentação inicial do projeto<br>
├── LICENSE                   # Licença do projeto<br>
├── .env                      # Arquivo para variáveis de ambiente (chaves de API, etc.)<br>
└── .gitignore                # Arquivo para ignorar arquivos/pastas no Git


---

### Pré-requisitos
Antes de começar, garanta que você tenha os seguintes softwares instalados:

- Python 3.11 ou superior

- Docker e Docker Compose

- Uma chave de API da OpenAI

---

### Configuração do Ambiente

Iniciar o Banco de Dados Vetorial (Milvus)

A maneira mais fácil de rodar o Milvus é via Docker. No terminal, na raiz do projeto, execute os seguintes comandos:

```Bash
# Baixar o arquivo de configuração do Milvus
wget https://milvus.io/docs/v2.4.x/assets/milvus/milvus-standalone-docker-compose.yml -O docker-compose.yml

# Iniciar os contêineres do Milvus em segundo plano
docker-compose up -d
```

Isso irá iniciar uma instância do Milvus pronta para uso em http://localhost:19530.

Configurar Variáveis de Ambiente

Crie um arquivo chamado .env na raiz do projeto e preencha-o com suas informações.

```Bash
# .env

# 1. Sua chave de API da OpenAI
# IMPORTANTE: Substitua pelo seu valor real
OPENAI_API_KEY="sk-..."

# 2. Configurações de conexão do Milvus
# Estes são os valores padrão para a instalação via Docker
MILVUS_AMB_URI="http://localhost:19530"
MILVUS_DB_NAME="default"
MILVUS_COLLECTION_NAME="owasp_asvs_v5"
```
Atenção: Mantenha seu arquivo .env seguro e nunca o compartilhe.

Criar Ambiente Virtual e Instalar Dependências

É altamente recomendado usar um ambiente virtual para isolar as bibliotecas do projeto.

```Bash
# Criar o ambiente virtual
python -m venv venv

# Ativar o ambiente (Windows)
.\venv\Scripts\activate

# Ativar o ambiente (macOS/Linux)
source venv/bin/activate

# Instalar todas as dependências
pip install -r requirements.txt
```

---

### Como Usar o Projeto
Com o ambiente configurado, siga esta sequência para processar seus dados e interagir com o agente.

Preparar os Documentos

- Crie uma pasta chamada data na raiz do projeto (se ela não existir).

- Coloque todos os documentos PDF que você deseja que o RAG utilize dentro desta pasta.

Executar o Parsing e a Ingestão

Estes dois comandos irão ler seus PDFs, processá-los e carregá-los no Milvus. Execute-os em ordem.

```Bash
# 1. Extrai o texto dos PDFs e cria o arquivo parsed_data.json
python parse_docs_to_json.py

# 2. Processa o JSON, cria os embeddings e armazena no Milvus
python ingestion.py
```

Este processo pode levar alguns minutos, dependendo do volume de documentos e do modelo de embedding utilizado.

(Opcional) Avaliar a Qualidade da Busca

Se você quiser medir a performance da estratégia de recuperação de dados, execute o script de avaliação. Ele usará o test_set.csv para fazer perguntas e um LLM para julgar a relevância dos resultados.

```Bash
python evaluate_retrieval.py
```

O resultado será exibido no terminal e salvo no arquivo evaluation_results.csv.

Interagir com o Agente RAG

Este é o passo final, onde você conversa com o assistente.

```Bash
python agent.py
```

O terminal exibirá a mensagem Agente RAG iniciado. Faça suas perguntas. Pressione Ctrl+C para sair.. Digite sua pergunta e pressione Enter. O agente irá raciocinar, buscar nos documentos e fornecer uma resposta completa com as fontes utilizadas.

Exemplo de pergunta:<br>
Sua Pergunta: Quais são os três níveis de verificação de segurança definidos pelo ASVS?

Para encerrar o agente, pressione ```Ctrl+C```.

Configuração Avançada (config.yaml)

O arquivo config.yaml permite customizar o comportamento do projeto sem alterar o código:

- ingestion_strategies: Defina diferentes estratégias de processamento de dados. Você pode variar o chunk_method (recursive ou semantic), o chunk_size, e o embedding_model. O partition_name isola os dados de cada estratégia no Milvus.

- evaluator: Configure o modelo LLM usado como juiz (llm_judge) e quantos documentos (retriever_k) ele deve avaliar.

- agent: Escolha qual strategy_to_use o agente principal deve utilizar, qual o seu modelo de LLM (agent_llm) e quantos documentos ele deve recuperar (retriever_k).

- retriever_models: Especifique os modelos de embedding e de re-ranking a serem utilizados pelo retriever_factory.