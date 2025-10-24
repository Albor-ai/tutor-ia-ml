# Plano de Desenvolvimento - Modelo de Recomendação de Conteúdo (CR)

**Objetivo Primário (Guiding Principle):** Utilizar o desenvolvimento do Modelo de Recomendação de Conteúdo como uma jornada de aprendizado em sistemas de recomendação, uma área especializada e muito relevante de Machine Learning.

**Objetivo Secundário (Técnico):** Desenvolver um modelo de recomendação que utilize o Perfil de Aprendizagem (LP) de um usuário para sugerir conteúdos relevantes, aprimorando a personalização da plataforma Tutor.IA.

**Visão Geral do Modelo CR:**
Este modelo funcionará como o "motor de descobertas" do Tutor.IA. Ele analisará um catálogo de conteúdos e o perfil do usuário para responder à pergunta: "Com base no que sabemos sobre este usuário, qual conteúdo ele gostaria de ver agora?".

**Abordagem Técnica: Filtragem Baseada em Conteúdo (Content-Based Filtering)**
Para este protótipo, usaremos uma abordagem de **Filtragem Baseada em Conteúdo**. A lógica é simples e poderosa: "Se um usuário gostou de um item, ele também gostará de itens *similares* a ele". Além disso, iremos adicionar uma camada de personalização, filtrando as recomendações para priorizar conteúdos que se alinhem com o Perfil de Aprendizagem (LP) do usuário.

---

### **Checklist de Desenvolvimento - Modelo de Recomendação de Conteúdo (CR)**

#### **Fase 1: Preparação e Estruturação dos Dados**

*   [ ] **1.1. Definir a Estrutura do Conteúdo:**
    *   **Objetivo:** Determinar quais são as características (metadados) que definem cada item de conteúdo em nossa plataforma.
    *   **Ação:** Criar uma lista de atributos para o nosso conteúdo.
    *   **Saída Esperada:** Definição de features como `content_id`, `content_type` (video, text, quiz), `topic` (python_basics, data_structures), `difficulty` (beginner, intermediate) e `associated_lp` (Visual, Auditivo, etc.).

*   [ ] **1.2. Definir a Estrutura da Interação do Usuário:**
    *   **Objetivo:** Definir como medimos o interesse ou consumo de um conteúdo por um usuário.
    *   **Ação:** Criar uma lista de atributos para as interações.
    *   **Saída Esperada:** Definição de features como `user_id`, `content_id`, `interaction_type` (completed, rated_highly) e o `learning_profile` do usuário.

*   [ ] **1.3. Gerar Dados Simulados:**
    *   **Objetivo:** Criar datasets sintéticos para o nosso catálogo de conteúdos e para as interações dos usuários.
    *   **Ação:** Desenvolver um script ou usar um notebook para gerar e salvar dois arquivos CSV.
    *   **Saída Esperada:**
        *   `data/raw/simulated_cr_content.csv`
        *   `data/raw/simulated_cr_interactions.csv`

*   [ ] **1.4. Análise Exploratória de Dados (EDA):**
    *   **Objetivo:** Entender as características dos nossos novos datasets.
    *   **Ação:** Criar um novo notebook (`notebooks/eda_cr_data.ipynb`) para explorar as distribuições de tipos de conteúdo, tópicos, e como os usuários interagem com eles.
    *   **Saída Esperada:** Um notebook com a análise e insights sobre os dados de conteúdo e interação.

#### **Fase 2: Pré-processamento e Modelagem de Conteúdo**

*   [ ] **2.1. Criar Notebook de Pré-processamento e Modelagem:**
    *   **Objetivo:** Centralizar as próximas etapas em um novo notebook.
    *   **Ação:** Criar `notebooks/modeling_cr_data.ipynb`.
    *   **Saída Esperada:** Notebook criado e dados carregados.

*   [ ] **2.2. Feature Engineering (Vetorização de Texto):**
    *   **Objetivo:** Converter os metadados de texto (como `topic` e `content_type`) em um formato numérico que possamos usar para cálculos matemáticos.
    *   **Ação:** Utilizar `TfidfVectorizer` do Scikit-learn para criar uma representação vetorial para cada item de conteúdo.
    *   **Aprendizado:** Entender como TF-IDF funciona para capturar a importância de palavras e como isso ajuda a definir a "essência" de um item.
    *   **Saída Esperada:** Uma matriz de features (vetores) onde cada linha representa um item de conteúdo.

*   [ ] **2.3. Cálculo da Matriz de Similaridade:**
    *   **Objetivo:** Calcular o quão "similar" cada item de conteúdo é de todos os outros.
    *   **Ação:** Usar a métrica de **Similaridade de Cossenos (Cosine Similarity)** na matriz de features gerada na etapa anterior.
    *   **Aprendizado:** Compreender o conceito de similaridade de cossenos e por que é eficaz para medir a similaridade entre vetores de texto.
    *   **Saída Esperada:** Uma matriz `(n_items x n_items)` onde cada célula `(i, j)` contém o score de similaridade entre o item `i` e o item `j`.

#### **Fase 3: Lógica de Recomendação e Validação**

*   [ ] **3.1. Desenvolver a Função de Recomendação:**
    *   **Objetivo:** Criar a lógica central que gera as recomendações para um usuário.
    *   **Ação:** Implementar uma função que:
        1.  Recebe um `user_id` e seu `learning_profile`.
        2.  Encontra os itens com os quais o usuário teve uma interação positiva.
        3.  Usa a matriz de similaridade para encontrar os itens mais similares aos que o usuário gostou.
        4.  Filtra e reordena os itens similares, dando um bônus para aqueles cujo `associated_lp` corresponde ao `learning_profile` do usuário.
        5.  Remove itens que o usuário já consumiu.
        6.  Retorna uma lista ordenada dos `N` melhores conteúdos para recomendar.
    *   **Saída Esperada:** Uma função de recomendação funcional.

*   [ ] **3.2. Validar a Lógica:**
    *   **Objetivo:** Testar a função de recomendação para garantir que ela se comporta como esperado.
    *   **Ação:** Executar a função para alguns usuários de teste com perfis diferentes e analisar a qualidade e a relevância das recomendações geradas.
    *   **Saída Esperada:** Confirmação de que a lógica está correta e faz sentido.

#### **Fase 4: Implantação do Protótipo (Deploy)**

*   [ ] **4.1. Salvar Artefatos:**
    *   **Objetivo:** Salvar os componentes necessários para fazer recomendações em produção.
    *   **Ação:** Salvar a matriz de similaridade e o `TfidfVectorizer` treinado usando `joblib` ou `pickle`.
    *   **Saída Esperada:** Arquivos de artefatos salvos em `src/models/`.

*   [ ] **4.2. Criar um Endpoint de Recomendação:**
    *   **Objetivo:** Expor a lógica de recomendação através de uma API.
    *   **Ação:** Adicionar um novo endpoint `/recommend/{user_id}` à nossa aplicação FastAPI (`main.py`) que chama a função de recomendação e retorna a lista de conteúdos recomendados.
    *   **Saída Esperada:** Um endpoint de API funcional para recomendações.
