# RAG
Assistente RAG especialista no OWASP ASVS. Desenvolvido com Python e LangChain, usa busca híbrida e re-ranking para respostas precisas sobre segurança de aplicações, baseando-se nos documentos. Inclui um sistema de avaliação para otimizar a recuperação da informação.

---

### ✨ Principais Características
- Fluxo RAG Completo: Scripts para pré-processamento de PDFs, ingestão de dados, avaliação de estratégias e execução do agente interativo.
- Recuperação Avançada: Utiliza busca híbrida (BM25 + FAISS) e um re-ranker Cross-Encoder para garantir a máxima relevância dos resultados.
- Módulo de Avaliação: Inclui um sistema "LLM-as-a-Judge" para testar e validar a eficácia de diferentes estratégias de chunking e embedding.
- Agente Inteligente: O agente final é construído com Chain of Thought para raciocinar sobre as perguntas e fornecer respostas precisas, citando as fontes e evidências do documento original.
- Configurável e Modular: Toda a lógica é controlada através de um arquivo config.yaml, facilitando a experimentação e a manutenção.

---

### ⚙️ Como Funciona
O projeto segue um fluxo de trabalho de RAG clássico, dividido em etapas claras:

- Extração: Os documentos PDF são lidos e seu conteúdo textual é extraído e limpo, gerando um arquivo JSON estruturado.
- Ingestão: O texto é dividido em pedaços (chunks) e vetorizado usando modelos de embedding. Esses vetores são armazenados em um banco de dados vetorial (FAISS) para busca semântica rápida.
- Recuperação: Quando um usuário faz uma pergunta, o sistema realiza uma busca híbrida (semântica + palavras-chave) para encontrar os chunks mais relevantes nos documentos.
- Geração: Os chunks recuperados são injetados em um prompt, junto com a pergunta do usuário, e enviados a um LLM (como o GPT-4o-mini) para gerar uma resposta coesa e contextualizada.

---

### 📋 Requisitos
Antes de começar, garanta que você tenha:
- Python 3.9 ou superior.
- Uma chave de API da OpenAI.
- Os documentos PDF que servirão como base de conhecimento do agente.

---

### 🚀 Instalação e Configuração
Siga estes passos para configurar o ambiente e preparar o projeto para execução.

1. Clonar o Repositório

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
```
```bash
cd seu-repositorio
```

2. Criar e Ativar o Ambiente Virtual
- Windows:

```bash
python -m venv venv
.\venv\Scripts\activate
```

- macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Instalar as Dependências
Com o ambiente virtual ativado, instale todas as bibliotecas necessárias:

```bash
pip install -r requirements.txt
```

4. Configurar a Chave de API
Crie um arquivo chamado .env na raiz do projeto e adicione sua chave da OpenAI:

```bash
OPENAI_API_KEY="sk-sua-chave-secreta-aqui"
```

5. Adicionar os Documentos
Coloque os seus arquivos PDF (por exemplo, o documento OWASP ASVS) dentro do diretório data/.

---

### ▶️ Execução
O projeto é executado em etapas. Siga a ordem abaixo.

Passo 1: Pré-processar os Documentos

Este script lê os PDFs da pasta data/, extrai o texto e cria o arquivo parsed_data.json.

```bash
python parse_docs_to_json.py
```

Passo 2: Ingerir os Dados (Criar Índices Vetoriais)

Este script processa o parsed_data.json e cria os bancos de dados vetoriais na pasta vector_stores/, um para cada estratégia definida em config.yaml.

```bash
python ingestion.py
```

Passo 3 (Opcional): Avaliar as Estratégias de Recuperação

Para determinar qual estratégia de ingestão oferece os melhores resultados, execute o script de avaliação. Ele usará o test_set.csv para pontuar cada estratégia e salvará os resultados em evaluation_results.csv.

```bash
python evaluate_retrieval.py
```

Após a execução, analise o .csv para decidir qual ID de estratégia usar no passo seguinte.

Passo 4: Executar o Agente Conversacional

Finalmente, para interagir com o AnalistaIA:

Edite o config.yaml: Abra o arquivo e, na seção agent, defina o valor de strategy_to_use para o ID da estratégia que você deseja usar (por exemplo, a que teve melhor pontuação na avaliação).

```bash
agent:
  strategy_to_use: 7 # <-- Altere este valor para o ID da melhor estratégia
  agent_llm: "gpt-4o-mini"
  retriever_k: 5
```

Inicie o agente:

```bash
python agent.py
```

Converse com o agente: O terminal exibirá o prompt Sua Pergunta:. Faça suas perguntas e pressione Enter. Para encerrar, pressione Ctrl+C.
