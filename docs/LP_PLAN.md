# Plano de Desenvolvimento - Modelo de Inferência do Perfil de Aprendizagem (LP) - 2025-10-23

**Objetivo Primário (Guiding Principle):** Utilizar o desenvolvimento do Modelo de Inferência do Perfil de Aprendizagem como uma jornada de aprendizado prática em Machine Learning e Deep Learning.

**Objetivo Secundário (Técnico):** Desenvolver um modelo de ML capaz de inferir o perfil de aprendizagem de um usuário no contexto do Tutor.IA, a fim de personalizar a experiência educacional.

**Visão Geral do Modelo LP:**
Este modelo é crucial para a personalização. Ele irá analisar dados de interação do usuário (como desempenho em exercícios, tempo gasto em diferentes tipos de conteúdo, preferências de recursos) para identificar ou classificar um "perfil de aprendizagem".

**O que é um "Perfil de Aprendizagem" (LP) para este projeto?**
Para começar de forma prática, vamos definir o Perfil de Aprendizagem como uma **classificação categórica**. Para este projeto, os perfis serão:
*   **Visual:** Alunos que se beneficiam mais de gráficos, diagramas e vídeos.
*   **Auditivo:** Alunos que compreendem melhor através de explicações verbais e áudios.
*   **Leitura/Escrita:** Alunos que preferem materiais escritos, como textos e notas, e gostam de fazer resumos.
*   **Cinestésico:** Alunos que aprendem "fazendo", através de exercícios práticos, simulações ou experimentos.

---

### **Checklist de Desenvolvimento - Modelo de Inferência LP**

Este checklist detalhado será nosso roteiro. Concluiremos as tarefas incrementalmente.

#### **Fase 1: Preparação e Entendimento Inicial** (Concluída)

*   [X] **1.1. Refinar a Definição do Perfil de Aprendizagem (LP):**
    *   **Objetivo:** Clarificar os tipos específicos de "perfis de aprendizagem" que o modelo tentará inferir.
    *   **Ação:** Decidir sobre uma lista de 3 a 5 categorias de perfis de aprendizagem. Isso guiará a criação do nosso "target" (o que o modelo prediz).
    *   **Saída Esperada:** Uma lista clara de perfis (ex: "Visual", "Auditivo", "Cinestésico").
    *   **Status:** Concluído. Perfis definidos como: Visual, Auditivo, Leitura/Escrita, Cinestésico.

*   [X] **1.2. Identificar Fontes de Dados Potenciais (Features):**
    *   **Objetivo:** Mapear quais informações sobre o usuário podem ser usadas como *features* (entradas) para o modelo.
    *   **Ação:** Listar exemplos concretos de dados de interação que o Tutor.IA poderia gerar, que indicariam um perfil de aprendizagem.
    *   **Saída Esperada:** Uma lista de 5-10 potenciais features, com uma breve descrição do que elas representam e seu tipo (numérico, categórico, etc.).
    *   **Status:** Concluído. Features identificadas:
        *   `time_spent_on_video` (Numérico)
        *   `time_spent_on_audio` (Numérico)
        *   `time_spent_reading` (Numérico)
        *   `time_spent_writing` (Numérico)
        *   `time_spent_on_quizz` (Numérico)
        *   `time_spent_on_flashcards` (Numérico)
        *   `completed_exercices` (Numérico)
        *   `completed_quizzes` (Numérico)
        *   `completed_flashcards` (Numérico)
        *   `text_quizzes_accuracy` (Numérico, 0-100)
        *   `visual_quizzes_accuracy` (Numérico, 0-100)
        *   `most_preferred_resource_type` (Categórico - simplificado de `most_preferred_resources` para simulação, e.g., 'video', 'audio', 'text', 'practical')

*   [X] **1.3. Estruturar o Ambiente de Desenvolvimento:**
    *   **Objetivo:** Preparar o ambiente de trabalho local para o desenvolvimento de ML.
    *   **Ação:**
        *   Verificar que um ambiente virtual Python (`.venv` ou equivalente) está ativo.
        *   Instalar as bibliotecas básicas necessárias: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn` (para visualização aprimorada).
        *   Criar uma estrutura de diretórios organizada no repositório para acomodar nossos artefatos (ex: `data/raw`, `data/processed`, `notebooks/`, `src/models/`, `src/utils/`).
    *   **Saída Esperada:** Ambiente de virtual configurado com dependências instaladas e estrutura de pastas criada.
    *   **Status:** Concluído. Diretórios criados manualmente pelo usuário.

#### **Fase 2: Coleta & Exploração de Dados (inicialmente com Dados Simulados)** (Concluída)

*   [X] **2.1. Criação de Dados Simulados para o Modelo LP:**
    *   **Objetivo:** Gerar um conjunto de dados sintéticos que imitem o tipo de informação que teríamos do Tutor.IA, incluindo os perfis de aprendizagem como rótulos.
    *   **Ação:** Utilizar `pandas` e `numpy` para criar um DataFrame com um número razoável de linhas (~1000 a 5000), contendo as `features` identificadas na etapa 1.2 e uma coluna `perfil_aprendizagem` (nosso `target`).
    *   **Saída Esperada:** Um arquivo `CSV` sintético salvo em `data/raw/simulated_lp_data.csv`.
    *   **Status:** Concluído. Dados simulados gerados manualmente pelo usuário.

*   [X] **2.2. Análise Exploratória de Dados (EDA) dos Dados Simulados:**
    *   **Objetivo:** Entender as características, padrões e distribuições dos dados que criamos.
    *   **Ação:** Criar um Jupyter Notebook (`notebooks/eda_lp_data.ipynb`) para:
        *   Carregar os dados.
        *   Verificar estatísticas descritivas (`.describe()`, `.info()`).
        *   Visualizar distribuições de features (histogramas, boxplots, gráficos de barras para categóricas, usando `matplotlib` e `seaborn`).
        *   Analisar a distribuição do `target` (`perfil_aprendizagem`).
        *   Explorar relações entre features e entre features e o target.
    *   **Saída Esperada:** Um notebook com a análise e os primeiros insights sobre os dados.
    *   **Status:** Concluído. Notebook `eda_lp_data.ipynb` criado. Agora você pode executá-lo e analisar os resultados.

#### **Fase 3: Pré-processamento e Feature Engineering** (Concluída)

*   [X] **3.1. Criar Notebook de Pré-processamento:**
    *   **Objetivo:** Iniciar um novo notebook focado nas etapas de pré-processamento dos dados.
    *   **Ação:** Criar `notebooks/preprocessing_lp_data.ipynb` e carregar o dataset `simulated_lp_data.csv`.
    *   **Saída Esperada:** Notebook `preprocessing_lp_data.ipynb` criado e dados carregados.
    *   **Status:** Concluído. Notebook `preprocessing_lp_data.ipynb` criado e corrigido.

*   [X] **3.2. Codificação do Target (`perfil_aprendizagem`):**
    *   **Objetivo:** Converter o target categórico (nomes dos perfis) em representação numérica que os modelos de ML podem usar.
    *   **Ação:** Utilizar `LabelEncoder` do Scikit-learn para transformar os 4 perfis de aprendizagem em números inteiros (0, 1, 2, 3).
    *   **Aprendizado:** Entender por que a codificação do target é necessária e os diferentes tipos de codificadores. `LabelEncoder` é adequado para o target de classificação.
    *   **Saída Esperada:** Uma nova coluna numérica para o target ou a transformação da coluna existente.
    *   **Status:** Concluído (implementado no notebook).

*   [X] **3.3. Codificação da Feature Categórica (`most_preferred_resource_type`):**
    *   **Objetivo:** Converter a feature categórica em representação numérica.
    *   **Ação:** Utilizar `OneHotEncoder` do Scikit-learn para transformar `most_preferred_resource_type` em colunas binárias.
    *   **Aprendizado:** Compreender a diferença entre `LabelEncoder` e `OneHotEncoder` e por que `OneHotEncoder` é geralmente preferível para features categóricas de entrada (para evitar a suposição de ordem ordinal).
    *   **Saída Esperada:** Novas colunas binárias no DataFrame, representando os tipos de recursos preferidos.
    *   **Status:** Concluído (implementado no notebook).

*   [X] **3.4. Escalamento das Features Numéricas:**
    *   **Objetivo:** Ajustar a escala das features numéricas para que nenhuma feature domine o treinamento do modelo indevidamente.
    *   **Ação:** Utilizar `StandardScaler` do Scikit-learn nas features numéricas.
    *   **Aprendizado:** Entender a importância do escalamento para algoritmos baseados em distância (como SVMs, KNN, redes neurais) e as diferenças entre `StandardScaler` e `MinMaxScaler`.
    *   **Saída Esperada:** Features numéricas transformadas com média zero e variância unitária.
    *   **Status:** Concluído (implementado no notebook).

*   [X] **3.5. Divisão dos Dados em Conjuntos de Treino e Teste:**
    *   **Objetivo:** Separar os dados em conjuntos para treinar o modelo e avaliar seu desempenho em dados nunca vistos.
    *   **Ação:** Utilizar `train_test_split` do Scikit-learn para dividir os dados pré-processados em 70% para treino e 30% para teste, garantindo estratificação pelo target.
    *   **Aprendizado:** Compreender a importância de conjuntos de treino/teste para evitar *overfitting* e a utilidade da estratificação em problemas de classificação.
    *   **Saída Esperada:** Variáveis `X_train`, `X_test`, `y_train`, `y_test`.
    *   **Status:** Concluído (implementado no notebook).

*   [X] **3.6. Salvar Dados Pré-processados:**
    *   **Objetivo:** Preservar o dataset pré-processado para uso posterior na fase de modelagem.
    *   **Ação:** Salvar os datasets `X_train`, `X_test`, `y_train`, `y_test` (ou `X_train` e `X_test` como DataFrames e `y_train`, `y_test` como Series/arrays) em `data/processed/`.
    *   **Saída Esperada:** Arquivos CSV ou formatos binários (como Parquet) para os dados de treino e teste.
    *   **Status:** Concluído (implementado no notebook e verificado pelo usuário).

*   [X] **4.1. Criar Notebook de Modelagem:**

    *   **Objetivo:** Iniciar um novo notebook focado nas etapas de modelagem.

    *   **Ação:** Criar `notebooks/modeling_lp_data.ipynb` e carregar os datasets `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` de `data/processed/`.

    *   **Saída Esperada:** Notebook `modeling_lp_data.ipynb` criado e dados carregados.

    *   **Status:** Concluído. Notebook `modeling_lp_data.ipynb` criado.



*   [X] **4.2. Selecionar Modelos Iniciais:**

    *   **Objetivo:** Escolher alguns algoritmos de classificação para começar. Para um problema de classificação multiclasse como este, boas opções iniciais são:

        *   **Regressão Logística (Logistic Regression):** Um modelo linear simples e um bom *baseline*.

        *   **Árvore de Decisão (Decision Tree Classifier):** Intuitivo e fácil de interpretar.

        *   **Random Forest Classifier:** Um *ensemble* de árvores de decisão, geralmente com bom desempenho.

    *   **Ação:** Importar esses modelos do Scikit-learn.

    *   **Aprendizado:** Entender os princípios básicos de cada algoritmo e por que são boas escolhas para começar.

    *   **Saída Esperada:** Modelos importados e prontos para uso.

    *   **Status:** Concluído (implementado no notebook).



*   [X] **4.3. Treinar Modelos:**

    *   **Objetivo:** Treinar os modelos selecionados usando os dados de treino.

    *   **Ação:** Instanciar e treinar cada modelo (`.fit(X_train, y_train)`).

    *   **Aprendizado:** Compreender o processo de treinamento e como os modelos aprendem com os dados.

    *   **Saída Esperada:** Modelos treinados.

    *   **Status:** Concluído (implementado no notebook).



*   [X] **4.4. Fazer Previsões:**

    *   **Objetivo:** Gerar previsões nos conjuntos de treino e teste para avaliar o desempenho.

    *   **Ação:** Utilizar `.predict(X_train)` e `.predict(X_test)` para cada modelo.

    *   **Aprendizado:** Entender a diferença entre prever no conjunto de treino (para verificar *overfitting*) e no conjunto de teste (para estimar o desempenho real).

    *   **Saída Esperada:** Previsões (`y_pred_train`, `y_pred_test`) para cada modelo.

    *   **Status:** Concluído (implementado no notebook).



*   [X] **4.5. Avaliar Modelos (Métricas Iniciais):**

    *   **Objetivo:** Quantificar o desempenho de cada modelo.

    *   **Ação:** Calcular `accuracy_score`, `precision_score`, `recall_score`, `f1_score` (com `average='weighted'` para multiclasse) e `confusion_matrix` para cada modelo nos dados de teste.

    *   **Aprendizado:** Compreender o significado de cada métrica de avaliação e por que são importantes para problemas de classificação multiclasse.

    *   **Saída Esperada:** Relatórios de métricas e matrizes de confusão para cada modelo.

    *   **Status:** Concluído (implementado no notebook).



*   [X] **4.6. Comparar Modelos:**

    *   **Objetivo:** Identificar qual modelo teve o melhor desempenho inicial.

    *   **Ação:** Comparar as métricas de avaliação dos modelos.

    *   **Saída Esperada:** Uma conclusão inicial sobre o modelo mais promissor.

    *   **Status:** Concluído (todos os modelos tiveram desempenho perfeito no dataset simulado).



#### **Fase 5: Avaliação** (Concluída)

*   [X] **5.1. Análise de Overfitting/Underfitting:**
    *   **Objetivo:** Verificar se os modelos estão aprendendo demais os dados de treino (overfitting) ou de menos (underfitting).
    *   **Ação:** Comparar as métricas de desempenho (Acurácia, F1-Score) entre os conjuntos de treino e teste.
    *   **Aprendizado:** Entender os conceitos de overfitting e underfitting e como identificá-los.
    *   **Saída Esperada:** Conclusão sobre o balanço entre bias e variância dos modelos.
    *   **Status:** Concluído (implementado no notebook).

*   [X] **5.2. Validação Cruzada (Cross-Validation):**
    *   **Objetivo:** Obter uma estimativa mais robusta do desempenho do modelo e reduzir a variância da avaliação.
    *   **Ação:** Implementar validação cruzada (e.g., K-Fold Cross-Validation) para os modelos, calculando a média e o desvio padrão das métricas.
    *   **Aprendizado:** Compreender a importância da validação cruzada para uma avaliação mais confiável do modelo.
    *   **Saída Esperada:** Métricas de desempenho com validação cruzada para cada modelo.
    *   **Status:** Concluído (implementado no notebook).

*   [X] **5.3. Curvas ROC e AUC (se aplicável):**
    *   **Objetivo:** Avaliar a capacidade dos modelos de distinguir entre as classes.
    *   **Ação:** Gerar curvas ROC e calcular a área sob a curva (AUC) para cada classe (abordagem one-vs-rest).
    *   **Aprendizado:** Entender o que são curvas ROC e AUC e sua relevância para avaliação de classificadores.
    *   **Saída Esperada:** Gráficos ROC e valores AUC.
    *   **Status:** Concluído (implementado no notebook e analisado).

#### **Fase 6 (Revisada): Finalização, Deploy e Próximos Passos**

*   [X] **6.1. Selecionar e Salvar o Melhor Modelo:**
    *   **Objetivo:** Escolher um dos modelos como nosso candidato "final" e salvar seu estado treinado em um arquivo `.joblib`.
    *   **Ação:** Usar a biblioteca `joblib` para salvar o objeto do modelo `RandomForestClassifier`.
    *   **Aprendizado:** Entender a importância de "persistir" um modelo.
    *   **Status:** Concluído. Modelo salvo em `src/models/lp_inference_model.joblib`.

*   [X] **6.2. Criar um Script de Inferência:**
    *   **Objetivo:** Desenvolver um script que carrega o modelo salvo e o utiliza para prever o perfil de um novo "usuário".
    *   **Ação:** Criar `src/predict.py` que carrega o modelo, aplica o pré-processamento e faz a previsão.
    *   **Aprendizado:** Compreender a diferença entre um pipeline de treino e um de inferência.
    *   **Status:** Concluído. Script `src/predict.py` criado e validado.

*   [X] **6.3. Criar um Endpoint de Inferência via API REST:**
    *   **Objetivo:** Expor o modelo de inferência através de uma API web.
    *   **Ação:** Utilizar o **FastAPI** para criar um endpoint `/predict` que recebe dados em JSON e retorna a previsão.
    *   **Aprendizado:** Aprender a servir um modelo de ML como um serviço web.
    *   **Status:** Concluído. API criada em `main.py` e validada.

*   [X] **6.4. Criar um Dashboard Interativo com Streamlit:**
    *   **Objetivo:** Criar uma interface de usuário simples para interagir com o modelo de forma visual.
    *   **Ação:** Desenvolver um script `app_streamlit.py` com um formulário dinâmico para prever o perfil de aprendizagem.
    *   **Aprendizado:** Aprender a construir rapidamente aplicações de dados interativas e a importância de salvar e reutilizar artefatos de pré-processamento (como o `StandardScaler`).
    *   **Status:** Concluído. Aplicação criada e validada.

*   [X] **6.5. Reflexão e Planejamento para Dados Reais:**
    *   **Objetivo:** Documentar o que foi feito e planejar os próximos passos para evoluir do protótipo para um sistema real.
    *   **Ação:** Atualizar o arquivo `docs/VISAO_FUTURA.md`.
    *   **Aprendizado:** Consolidar o conhecimento de que dados simulados são um ponto de partida, mas o mundo real tem mais complexidades.
    *   **Status:** Concluído. Documento de visão futura atualizado com o resumo do projeto e próximos passos.

#### **Fase 7: Otimização de Hiperparâmetros** (Concluída)

*   [X] **7.1. Definir o Espaço de Busca (Grid de Hiperparâmetros):**
    *   **Objetivo:** Definir quais hiperparâmetros do Random Forest vamos ajustar e quais valores testar.
    *   **Ação:** Criar um dicionário (grid) com os hiperparâmetros, como `n_estimators` (número de árvores), `max_depth` (profundidade máxima), `min_samples_split`, etc.
    *   **Aprendizado:** Entender o que são hiperparâmetros e como eles impactam o comportamento e o desempenho do modelo.
    *   **Status:** Concluído.

*   [X] **7.2. Escolher a Estratégia de Busca:**
    *   **Objetivo:** Selecionar um método eficiente para encontrar a melhor combinação de hiperparâmetros.
    *   **Ação:** Utilizar `RandomizedSearchCV` do Scikit-learn. É mais rápido que `GridSearchCV` e muito eficaz para uma primeira otimização.
    *   **Aprendizado:** Compreender a diferença entre busca aleatória (`RandomizedSearchCV`) e busca em grade (`GridSearchCV`).
    *   **Status:** Concluído.

*   [X] **7.3. Executar a Otimização:**
    *   **Objetivo:** Rodar o processo de busca para encontrar os melhores hiperparâmetros.
    *   **Ação:** Instanciar e executar o `.fit()` no objeto `RandomizedSearchCV` com os dados de treino.
    *   **Saída Esperada:** Um objeto treinado contendo os resultados da busca.
    *   **Status:** Concluído.

*   [X] **7.4. Avaliar o Melhor Modelo Encontrado:**
    *   **Objetivo:** Analisar os resultados da otimização e verificar se houve melhoria.
    *   **Ação:** Extrair os melhores parâmetros (`.best_params_`) e o melhor score (`.best_score_`). Treinar um novo modelo com esses parâmetros e avaliar no conjunto de teste.
    *   **Saída Esperada:** Uma comparação do desempenho do modelo otimizado versus o modelo baseline.
    *   **Status:** Concluído. A otimização não resultou em melhoria de desempenho no conjunto de teste (0.00% de ganho), indicando que o gargalo para este dataset não está nos hiperparâmetros do Random Forest.

#### **Fase 8: Experimentação com Novos Modelos (XGBoost)** (Concluída)

*   [X] **8.1. Instalar a Biblioteca XGBoost:**
    *   **Objetivo:** Adicionar o `XGBoost`, uma biblioteca de Gradient Boosting de alta performance, ao nosso ambiente.
    *   **Ação:** Executar `uv pip install xgboost` e `brew install libomp` (no macOS) para instalar a biblioteca e suas dependências.
    *   **Aprendizado:** Como gerenciar dependências e resolver problemas de bibliotecas externas (`libomp`).
    *   **Status:** Concluído.

*   [X] **8.2. Adicionar XGBoost ao Pipeline de Treinamento:**
    *   **Objetivo:** Integrar o `XGBClassifier` ao nosso notebook de modelagem para uma comparação direta.
    *   **Ação:** Importar e adicionar o `XGBClassifier` ao dicionário `models` no notebook.
    *   **Aprendizado:** A vantagem de ter um pipeline de experimentação que permite testar novos modelos facilmente.
    *   **Status:** Concluído.

*   [X] **8.3. Otimizar e Comparar o Desempenho:**
    *   **Objetivo:** Otimizar os hiperparâmetros do XGBoost e avaliar se ele supera o baseline do Random Forest.
    *   **Ação:** Usar `RandomizedSearchCV` para encontrar a melhor combinação de hiperparâmetros para o XGBoost.
    *   **Saída Esperada:** Uma conclusão clara se o XGBoost oferece um desempenho superior.
    *   **Status:** Concluído. A otimização resultou em uma **melhora de 3.91%** em relação ao XGBoost baseline. O modelo otimizado alcançou **66.50% de acurácia**, tornando-se o nosso novo campeão. Os melhores parâmetros (`max_depth=3`, `learning_rate=0.01`) indicam que um modelo que aprende de forma mais lenta e com árvores mais simples generaliza melhor para este problema.

#### **Fase 9: Finalização e Deploy do Modelo** (Concluída)

*   [X] **9.1. Salvar o Modelo Final:**
    *   **Objetivo:** Persistir o nosso melhor modelo (XGBoost otimizado) para que possa ser usado em produção sem a necessidade de retreinamento.
    *   **Ação:** Treinar o `XGBClassifier` com os melhores hiperparâmetros em **todos** os dados de treino e salvar o objeto do modelo em um arquivo `lp_model.joblib` usando a biblioteca `joblib`.
    *   **Aprendizado:** Como salvar e versionar modelos treinados.
    *   **Status:** Concluído. O pipeline completo, incluindo o modelo e os pré-processadores, foi salvo em `src/models/lp_model.joblib`.

*   [X] **9.2. Criar um Script de Inferência:**
    *   **Objetivo:** Desenvolver um script que simule o uso do modelo em um ambiente de produção.
    *   **Ação:** Criar um script `predict.py` que:
        1.  Carrega o pipeline e o `LabelEncoder` salvos.
        2.  Recebe dados brutos de um novo usuário.
        3.  Usa o pipeline para fazer uma previsão e decodifica o resultado.
    *   **Aprendizado:** A importância de um pipeline de inferência consistente que replica o ambiente de treinamento, usando artefatos salvos para evitar erros.
    *   **Status:** Concluído. O script `predict.py` foi criado e validado, prevendo corretamente o perfil de um usuário de exemplo.

---

### **Conclusão do Projeto**

O projeto foi concluído com sucesso, passando por todas as etapas de um ciclo de vida de Machine Learning, desde a concepção e geração de dados até a experimentação, otimização e criação de um pipeline de inferência funcional. O modelo final, um `XGBoost` otimizado, foi salvo e está pronto para ser integrado em aplicações maiores.
